# %%
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy
import pandas
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from numpy import ndarray
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from cath.cath_shared import (
    figures,
    remapped_fasta,
    fasta_file,
    load_files,
    cath_data,
    cath_dbs,
    load_mapping,
)
from seqvec_search import mmseqs
from seqvec_search.utils import E_VALUE_CUTOFF, endfig, rolling_mean

# %%

load_files()

# %%

print("Loading hits and scores")
metric = "cosine"
assert cath_data.joinpath(f"hits_{metric}.npz").is_file()
assert cath_data.joinpath(f"scores_{metric}.npz").is_file()
hits_per_method = dict(numpy.load(cath_data.joinpath(f"hits_{metric}.npz")))
scores_per_method = dict(numpy.load(cath_data.joinpath(f"scores_{metric}.npz")))

ids = numpy.asarray(
    [
        i.split("|")[2].split("/")[0]
        for i in json.loads(cath_data.joinpath("ids.json").read_text())
    ]
)

mapping_levels, mapping_array = load_mapping(ids)


# %%


def print_level_stats():
    for pos, level in zip(reversed(range(4)), "CATH"):
        counter = Counter(i[pos] for i in mapping_levels.values())
        print(level, sorted(counter.items(), reverse=True, key=lambda x: x[1])[:5])
        data = numpy.asarray([counter[value[pos]] for value in mapping_levels.values()])

        print(
            f"There are {len(data)} entries, of which {sum(data > 1)} have at least one match on level {level}"
        )


print_level_stats()


# %%

# All against all, check on all levels
# Note: We check whether the second hit matches, since the first is the self hit


def compute_is_correct(results: ndarray) -> ndarray:
    # queries -> levels -> hits
    # noinspection PyUnresolvedReferences
    return numpy.asarray(
        [
            (mapping_array[query] == mapping_array[result]).T
            for query, result in zip(range(len(results)), results)
        ]
    )


print("Checking hits for correctness")
is_correct_all_per_method = {
    name: compute_is_correct(results) for name, results in hits_per_method.items()
}

# Make a possibility map for is_correct

family_sizes = [
    Counter(family_name[level] for family_name in mapping_levels.values())
    for level in range(4)
]

# We only consider those that have more than one possible hit on the class level
is_possible = numpy.asarray([family_sizes[0][mapping_levels[id][0]] > 1 for id in ids])

# This map has a) 0 for every family with only 1 member and b) otherwise the weight to correct the
# accuracy by superfamily size, i.e. 1/superfamily size. We can then compute the normalized accuracy
# by summing up and dividing by the superfamily size
normalization = numpy.asarray(
    [1 / family_sizes[0][mapping_levels[id][0]] for id in ids]
)
normalization[~is_possible] = 0
families_count = sum(size > 1 for size in family_sizes[0].values())

is_correct_per_method = {
    name: is_correct_[is_possible, 0, 0]
    for name, is_correct_ in is_correct_all_per_method.items()
}
ids_possible = ids[is_possible]

# %%

print(
    f"Families : {len(family_sizes[0].values())}, "
    f"Families >1 member: {len([i for i in family_sizes[0].values() if i > 1])}"
)

# %%

# Since we know this is the best one, we'll only do the further validations with seqvec cosine
best_tag = "ProtT5 XL U50"
best_tag_short = "ProtT5"
is_correct_all = is_correct_all_per_method[best_tag]
is_correct_best = is_correct_per_method[best_tag]
scores_best = scores_per_method[best_tag][is_possible, 0]
best_name = "ProtT5 XL U50"
e = 0.01  # E-value cutoff: Below MMseqs2, above knn

# %%

# Compare to MMseqs2


def run_mmseqs(
    db_name: Path, out_db: Path, e_value: float = E_VALUE_CUTOFF
) -> Tuple[Dict[int, ndarray], Dict[int, ndarray]]:
    if not remapped_fasta.is_file():
        # We don't want to have any id mismatches
        assert numpy.array_equal(
            ids, [i[12:19] for i in fasta_file.read_text().splitlines()[::2]]
        )
        with remapped_fasta.open("w") as fp:
            for id, sequence in zip(ids, fasta_file.read_text().splitlines()[1::2]):
                fp.write(f">{id}\n{sequence}\n")

    if not out_db.with_suffix(".dbtype").is_file():
        mmseqs.create_db(remapped_fasta, db_name)

        with TemporaryDirectory() as temp_dir:
            subprocess.check_call(
                [
                    "mmseqs/bin/mmseqs",
                    "search",
                    "-e",
                    str(e_value),
                    "-s",
                    str(7.5),
                    str(db_name),
                    str(db_name),
                    str(out_db),
                    temp_dir,
                ]
            )

    mmseqs_hits_, mmseqs_hits_e_all_e_values_ = mmseqs.read_result_db_with_e_value(
        list(ids), db_name, list(ids), db_name, out_db
    )

    # Remove the self hit
    mmseqs_hits_e_all_e_values_ = {
        hit_id: hits[1:] for hit_id, hits in mmseqs_hits_e_all_e_values_.items()
    }
    mmseqs_hits_ = {id: hits[1:] for id, hits in mmseqs_hits_.items()}
    return mmseqs_hits_, mmseqs_hits_e_all_e_values_


mmseqs_hits, mmseqs_hits_e_all_e_values = run_mmseqs(
    cath_dbs.joinpath("cath20"), cath_dbs.joinpath("out")
)

# mmseqs_hits, mmseqs_hits_e_all_e_values = mmseqs.read_result_db_with_e_value(
#    list(ids),
#    cath_dbs.joinpath("cath20"),
#    list(ids),
#    cath_dbs.joinpath("cath20"),
#    Path("/home/schuetze/uniref30_expandaln/res_cath_profile_cath"),
# )

# No hit: 11

# %%


def evaluate_mmseqs(
    mmseqs_hits_: Dict[int, ndarray], mmseqs_hits_e_all_e_values_: Dict[int, ndarray]
) -> Tuple[ndarray, ndarray]:
    # This has only one hit instead of 5, but otherwise works identically

    no_hit = 0
    is_correct_mmseqs_ = []
    e_value_hits_ = []
    for query_num, id in enumerate(ids):
        if len(mmseqs_hits_[query_num]) == 0:
            # No hit found
            no_hit += 1
            is_correct_mmseqs_.append(numpy.asarray([[False]] * 4))
            e_value_hits_.append(10**6)
            continue
        is_correct_mmseqs_.append(
            numpy.asarray(
                [
                    [
                        mapping_levels[id][i]
                        == mapping_levels[ids[mmseqs_hits_[query_num][0]]][i]
                    ]
                    for i in range(4)
                ]
            )
        )
        e_value_hits_.append(mmseqs_hits_e_all_e_values_[query_num][0])

    print(f"No hit: {no_hit}")
    return numpy.asarray(is_correct_mmseqs_), numpy.asarray(e_value_hits_)


is_correct_all_mmseqs, e_value_hits = evaluate_mmseqs(
    mmseqs_hits, mmseqs_hits_e_all_e_values
)
is_correct_all_per_method["MMseqs2"] = is_correct_all_mmseqs

e_values_cath = e_value_hits[is_possible]
is_correct_all_per_method[f"MMseqs2 E<{e}"] = is_correct_all_mmseqs.copy()
is_correct_all_per_method[f"MMseqs2 E<{e}"][e_value_hits > e] = numpy.asarray(
    [[False]] * 4
)


# %%

# Check class imbalance
# We don't filter for is possible since all are always possible


def check_class_imbalance():
    klasses = numpy.asarray(
        [mapping_levels[ids[i]][3] for i in range(len(is_correct_all))]
    )

    print("class imbalance")
    print("| C | A | T | H |")
    print("|:---:|:---:|:---:|")

    print(
        " | ".join(
            str(is_correct_all[(klasses == str(klass)) & is_possible].shape[0])
            for klass in range(1, 5)
        )
    )

    print(
        " | ".join(
            str(is_correct_all[klasses == str(klass)].shape[0]) for klass in range(1, 5)
        )
    )

    for level in [0, 3]:
        for is_correct_all_ in [is_correct_all, is_correct_all_mmseqs]:
            print(
                level,
                " | ".join(
                    "{:.1%}".format(
                        float(
                            is_correct_all_[(klasses == str(klass)) & is_possible][
                                :, level, 0
                            ].sum()
                            / is_correct_all_[
                                (klasses == str(klass)) & is_possible
                            ].shape[0]
                        )
                    )
                    for klass in range(1, 5)
                ),
            )


check_class_imbalance()

# %%

family_correct_best = defaultdict(int)
for query, correct in zip(range(len(is_correct_all)), is_correct_all[:, 0, 0]):
    family_correct_best[mapping_levels[ids[query]][0]] += 1 if correct else 0

family_correct_mmseqs = defaultdict(int)
for query, correct in zip(
    range(len(is_correct_all_mmseqs)), is_correct_all_mmseqs[:, 0, 0]
):
    family_correct_mmseqs[mapping_levels[ids[query]][0]] += 1 if correct else 0

data = dict()
for label, family_correct_ in [
    (best_name, family_correct_best),
    ("MMseqs2", family_correct_mmseqs),
]:
    points = [
        (family_sizes[0][key], family_correct_[key] / family_sizes[0][key])
        for key in family_correct_.keys()
    ]
    data[label] = list(zip(*points))

numpy.savez(figures.joinpath("superfamily-vs-accuracy.npz"), **data)

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
for label, points in data.items():
    plt.scatter(*points, s=4, label=label)
plt.xlabel("Superfamily size")
plt.ylabel("QrawTop1")
plt.legend()
plt.grid()
endfig(figures, "superfamily-vs-accuracy")

# %%

# Confusion matrix
#                 SeqVec correct   SeqVec wrong
# MMseqs correct            2399           1987
# MMseqs wrong              1679           4809

print(f"{'':>15}{best_name}{' correct':>15}{best_name}{' wrong':>15}")

is_correct_mmseqs = is_correct_all_mmseqs[is_possible, 0, 0]
print(
    f"MMseqs correct {(is_correct_best & is_correct_mmseqs).sum():>15}{(~is_correct_best & is_correct_mmseqs).sum():>15}"
)
print(
    f"MMseqs wrong   {(is_correct_best & ~is_correct_mmseqs).sum():>15}{(~is_correct_best & ~is_correct_mmseqs).sum():>15}"
)

# %%

# Combine MMseqs2 and best cosine

is_correct_combined = []
for query_num, id in enumerate(ids):
    if len(mmseqs_hits[query_num]) == 0 or e_value_hits[query_num] > e:
        # No hit found
        is_correct_combined.append(is_correct_all[query_num][:, :1])
        continue
    is_correct_combined.append(
        [
            [mapping_levels[id][i] == mapping_levels[ids[mmseqs_hits[query_num][0]]][i]]
            for i in range(4)
        ]
    )

is_correct_combined = numpy.asarray(is_correct_combined)

records = [
    (
        best_name,
        is_correct_best.mean(),
        (is_correct_all[:, 0, 0] * normalization).sum() / families_count,
    ),
    (
        "MMseqs2",
        is_correct_mmseqs.mean(),
        (is_correct_all_mmseqs[:, 0, 0] * normalization).sum() / families_count,
    ),
    (
        f"MMseqs2 E<{e}",
        is_correct_all_per_method[f"MMseqs2 E<{e}"][is_possible, 0].mean(),
        (is_correct_all_per_method[f"MMseqs2 E<{e}"][:, 0, 0] * normalization).sum()
        / families_count,
    ),
    (
        f"MMseqs2 E<{e} + knnProtT5",
        is_correct_combined[is_possible, 0].mean(),
        (is_correct_combined[:, 0, 0] * normalization).sum() / families_count,
    ),
    (
        "Perfect merger",
        (is_correct_best | is_correct_mmseqs).mean(),
        ((is_correct_all | is_correct_all_mmseqs)[:, 0, 0] * normalization).sum()
        / families_count,
    ),
]
print(
    pandas.DataFrame.from_records(records, columns=["Method", "QrawTop1", "QnormTop1"])
    .set_index("Method")
    .to_markdown(floatfmt=".1%", tablefmt="plain")
)
is_correct_all_per_method[f"MMseqs2 E<{e} + {best_name}"] = is_correct_combined


# %%


def bootstrap_scores(is_correct: ndarray, normalized: float) -> Tuple[float, float]:
    rng = numpy.random.default_rng(42)
    bootstrapped_normalized = []
    bootstrapped_raw = []
    for _ in tqdm(range(500)):
        sample_ids = rng.choice(len(is_correct), len(is_correct))
        smol_mapping = mapping_array[is_possible, 0][sample_ids]
        assert smol_mapping.shape == sample_ids.shape
        family_counts = Counter(smol_mapping)
        normalization_array = numpy.asarray(
            [1 / family_counts[family] for family in smol_mapping]
        )
        bootstrapped_normalized.append(
            (is_correct[sample_ids] * normalization_array).sum() / len(family_counts)
        )
        bootstrapped_raw.append(is_correct[sample_ids].mean())
    bootstrapped_raw = numpy.asarray(bootstrapped_raw)
    bootstrapped_raw.sort()
    bootstrapped_normalized = numpy.asarray(bootstrapped_normalized)
    bootstrapped_normalized.sort()

    normalized_lower = bootstrapped_normalized[
        int(len(bootstrapped_normalized) * 0.025)
    ]
    normalized_upper = bootstrapped_normalized[
        int(len(bootstrapped_normalized) * 0.975)
    ]
    plus_minus_normalized = max(
        normalized - normalized_lower, normalized_upper - normalized
    )

    raw_lower = bootstrapped_raw[int(len(bootstrapped_raw) * 0.025)]
    row_upper = bootstrapped_raw[int(len(bootstrapped_raw) * 0.975)]
    plus_minus_raw = max(is_correct.mean() - raw_lower, row_upper - is_correct.mean())
    return plus_minus_normalized, plus_minus_raw


def make_report_accuracies_table(
    is_correct_all_per_method_: Dict[str, ndarray], bootstrap: bool
) -> pandas.DataFrame:
    class_accuracies = dict()
    for name, is_correct in is_correct_all_per_method_.items():
        normalized = (is_correct[:, 0, 0] * normalization).sum() / families_count
        raw = (is_correct[is_possible, 0, 0]).mean()
        if bootstrap:
            class_accuracies[name] = (normalized, raw) + bootstrap_scores(
                is_correct[is_possible, 0, 0], normalized
            )
        else:
            class_accuracies[name] = (normalized, raw)
    if bootstrap:
        index = ["normalized", "raw", "normalized_stderr", "raw_stderr"]
    else:
        index = ["normalized", "raw"]
    class_accuracies = pandas.DataFrame(class_accuracies, index=index).T
    class_accuracies = class_accuracies.sort_values("normalized", ascending=False)
    if bootstrap:
        raw = (
            class_accuracies["raw"].apply("{:.1%}".format)
            + "±"
            + class_accuracies["raw_stderr"].apply("{:.1%}".format)
        )
        normalized = (
            class_accuracies["normalized"].apply("{:.1%}".format)
            + "±"
            + class_accuracies["normalized_stderr"].apply("{:.1%}".format)
        )
    else:
        raw = class_accuracies["raw"].apply("{:.1%}".format)
        normalized = class_accuracies["normalized"].apply("{:.1%}".format)
    report_accuracies_table = pandas.DataFrame({"normalized": normalized, "raw": raw})
    return report_accuracies_table


def report_accuracies(is_correct_all_per_method_: Dict[str, ndarray]):
    print(f"Dataset size: {len(is_possible)} {is_possible.sum()}")
    report_accuracies_table = make_report_accuracies_table(
        is_correct_all_per_method_, False
    )
    if metric == "cosine":
        accuracies_filenames = "accuracies.md"
    else:
        hits_per_method_cosine = dict(
            numpy.load(cath_data.joinpath(f"hits_cosine.npz"))
        )
        is_correct_all_per_method_cosine = {
            name: compute_is_correct(results)
            for name, results in hits_per_method_cosine.items()
        }
        report_accuracies_table_cosine = make_report_accuracies_table(
            is_correct_all_per_method_cosine, False
        )

        report_accuracies_table = report_accuracies_table_cosine.join(
            report_accuracies_table, rsuffix=" euclidean", lsuffix=" cosine"
        )
        report_accuracies_table = report_accuracies_table[
            ["normalized euclidean", "normalized cosine", "raw euclidean", "raw cosine"]
        ]

        accuracies_filenames = "accuracies_euclidean.md"
    print(report_accuracies_table.to_string())
    with figures.joinpath(accuracies_filenames).open("w") as fp:
        report_accuracies_table.to_markdown(fp)

    header = list("CATH")
    df = pandas.DataFrame(
        {
            name: [
                (is_correct[:, level, 0] * normalization).sum() / families_count
                for level in reversed(range(4))
            ]
            for name, is_correct in is_correct_all_per_method_.items()
        },
        index=header,
    ).T
    df = df.sort_values(header[-1], ascending=False)
    # https://stackoverflow.com/a/31671975/3549270
    print(df.to_string(float_format="{:.1%}".format))
    if metric == "cosine":
        df.to_csv(figures.joinpath("normalized_scores.csv"))

    df = pandas.DataFrame(
        {
            name: [
                (is_correct[is_possible, level, 0]).mean()
                for level in reversed(range(4))
            ]
            for name, is_correct in is_correct_all_per_method_.items()
        },
        index=header,
    ).T
    df = df.sort_values(header[-1], ascending=False)
    if metric == "cosine":
        df.to_csv(figures.joinpath("raw_scores.csv"))

    print(f"{'Levels':<27}" + "".join("{:>12}".format(i) for i in header))
    is_correct_items = list(is_correct_all_per_method_.items())
    is_correct_items.sort(key=lambda x: -x[1][is_possible, 0].mean())
    for name, is_correct in is_correct_items:
        accuracies = [
            "{:>7.1%}".format(is_correct[is_possible, level, 0].mean())
            for level in reversed(range(4))
        ]

        stderrs = [
            "±{:.1%}".format(
                is_correct[is_possible, level, 0].std()
                / numpy.sqrt(len(is_correct[is_possible, level, 0]))
            )
            for level in reversed(range(4))
        ]

        line = "".join(a + b for a, b in zip(accuracies, stderrs))

        # Show overview
        print(f"{name:<27}{line}")


report_accuracies(is_correct_all_per_method)

# %%

points = [
    (
        family_sizes[0][key],
        (family_correct_best[key] / family_sizes[0][key])
        - (family_correct_mmseqs[key] / family_sizes[0][key]),
    )
    for key in family_correct_.keys()
]

points = numpy.asarray(list(zip(*points)))
numpy.savez(figures.joinpath("superfamily-vs-delta-accuracy.npy"), points)

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
plt.scatter(*points)
# plt.title(f"Superfamily size vs. how much more accurate {best_name} is")
plt.xlabel("Superamily size")
plt.ylabel(f"QrawTop1 {best_name} - QrawTop1 MMseqs2")
plt.grid()
plt.hlines(0, 0, 250, colors="black")
endfig(figures, "superfamily-vs-delta-accuracy")


# %%


def plot_accuracy_combined(
    mmseqs_top_: float,
    knn_: float,
    e_accuracy_simple_: List[float],
    e_accuracy_combined_: List[float],
    name: str,
    y_label: str,
):
    numpy.savez(
        figures.joinpath(name + ".npz"),
        e_accuracy_simple_=e_accuracy_simple_,
        e_accuracy_combined_=e_accuracy_combined_,
    )

    plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
    plt.plot(
        x_axis,
        e_accuracy_combined_,
        label=f"MMseqs2 with cutoff + knn{best_tag_short}",
    )
    plt.axhline(knn_, color="green", label=f"knn{best_tag_short}")
    plt.axhline(mmseqs_top_, color="black", label="MMseqs2 baseline")
    plt.plot(x_axis, e_accuracy_simple_, label="MMseqs2 with cutoff")
    plt.xscale("log")
    plt.ylim((0, 1))
    plt.xlabel("E-Value cutoff")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    # plt.title(f"QrawTop1 of MMseqs2 and {best_name} combined by E-Value cutoff")
    endfig(figures, name)


e_value_stretched = e_value_hits[..., numpy.newaxis, numpy.newaxis]

x_axis = numpy.logspace(-10, 2, 50)

e_accuracy_combined = list()
e_accuracy_simple = list()
for e_value in tqdm(x_axis):
    e_accuracy_combined.append(
        (
            ((e_value_stretched < e_value) & is_correct_all_mmseqs)
            | ((e_value_stretched > e_value) & is_correct_all)
        )[is_possible, 0, 0].mean()
    )

    e_accuracy_simple.append(
        ((e_value_stretched < e_value) & is_correct_all_mmseqs)[
            is_possible, 0, 0
        ].mean()
    )
plot_accuracy_combined(
    is_correct_mmseqs.mean(),
    is_correct_best.mean(),
    e_accuracy_simple,
    e_accuracy_combined,
    "combining-mmseqs-and-knn-raw",
    "QrawTop1",
)

# %%

plt.plot(
    is_correct_best[numpy.argsort(-scores_best)].cumsum() / len(is_correct_best),
    label="knnProtT5 cosine",
)
plt.plot(
    is_correct_best[numpy.argsort(e_value_hits[is_possible])].cumsum()
    / len(is_correct_best),
    label="knnProtT5 E-value",
)
plt.plot(
    is_correct_all_mmseqs[is_possible, 0, 0][numpy.argsort(-scores_best)].cumsum()
    / len(is_correct_best),
    label="MMseqs2 cosine",
)
plt.plot(
    is_correct_all_mmseqs[is_possible, 0, 0][
        numpy.argsort(e_value_hits[is_possible])
    ].cumsum()
    / len(is_correct_best),
    label="MMseqs2 E-value",
)
plt.title("Comparing different ways if sorting hits (cumulative plot)")
plt.ylabel("raw accuracy")
plt.xlabel("number of queries annotated")
plt.ylim((0, 1))
plt.grid()
plt.legend()
plt.show()

# %%


e_accuracy_combined = list()
e_accuracy_simple = list()
for e_value in tqdm(x_axis):
    e_accuracy_combined.append(
        (
            (
                ((e_value_stretched < e_value) & is_correct_all_mmseqs)
                | ((e_value_stretched > e_value) & is_correct_all[:, :, :1])
            )[:, 0, 0]
            * normalization
        ).sum()
        / families_count
    )

    e_accuracy_simple.append(
        (
            ((e_value_stretched < e_value) & is_correct_all_mmseqs)[:, 0, 0]
            * normalization
        ).sum()
        / families_count
    )

mmseqs_top = (is_correct_all_mmseqs[:, 0, 0] * normalization).sum() / families_count
k_nn_top = (is_correct_all[:, 0, 0] * normalization).sum() / families_count
plot_accuracy_combined(
    mmseqs_top,
    k_nn_top,
    e_accuracy_simple,
    e_accuracy_combined,
    "combining-mmseqs-and-knn-normalized",
    "QnormTop1",
)

# %%

e = 0.01
# With the E-value cutoff, the log makes sure the E-value scores are always below the cosine scores
combined_scores = numpy.log(e_values_cath.copy())
combined_scores[e_values_cath > e] = -scores_best[e_values_cath > e]
is_correct_combined = is_correct_mmseqs.copy()
is_correct_combined[e_values_cath > e] = is_correct_best[e_values_cath > e]

fig, ax = plt.subplots(figsize=(5, 5 * (4.8 / 6.4)))

# secax = ax.secondary_xaxis('top', functions=(lambda x: int(e_values_cath[int(len(e_values_cath) * x)]), lambda x: numpy.sum(e_values_cath < x)))
# secax.set_xlabel('E-value cutoff')


plt.plot(
    numpy.linspace(0, 1, len(is_correct_best)),
    (
        numpy.cumsum(is_correct_best[numpy.argsort(-scores_best)])
        / (len(is_correct_best) + 1)
    ),
    label=f"knn{best_tag_short}",
)
plt.plot(
    numpy.linspace(0, 1, len(is_correct_mmseqs))[numpy.sum(e_values_cath < e) :],
    (
        numpy.cumsum(is_correct_mmseqs[numpy.argsort(e_values_cath)])
        / (len(is_correct_best) + 1)
    )[numpy.sum(e_values_cath < e) :],
    label="MMseqs2",
)
plt.plot(
    numpy.linspace(0, 1, len(is_correct_combined))[numpy.sum(e_values_cath < e) :],
    (
        numpy.cumsum(is_correct_combined[numpy.argsort(combined_scores)])
        / (len(is_correct_best) + 1)
    )[numpy.sum(e_values_cath < e) :],
    label=f"MMseqs2 E<{e} + knn{best_tag_short}",
)
plt.xlabel("Fraction of annotated queries")
plt.ylabel("QrawTop1")

plt.plot([0, 1], [0, 1], color="grey", linestyle="dashed", label="Perfect method")

plt.legend()
plt.grid()

# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
x = numpy.linspace(0, numpy.mean(e_values_cath < e), numpy.sum(e_values_cath < e))
y = numpy.cumsum(
    is_correct_combined[numpy.argsort(combined_scores)][: numpy.sum(e_values_cath < e)]
) / (len(is_correct_best) + 1)

points = numpy.array([x, y]).T.reshape(-1, 1, 2)
segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
cmap = ListedColormap(plt.rcParams["axes.prop_cycle"].by_key()["color"][1:3])
seg_width = int(len(x) / 25)
norm = BoundaryNorm([0, seg_width / 2, seg_width], 2)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array((numpy.arange(len(points)) % seg_width))
lc.set_linewidth(2)
ax.add_collection(lc)

plt.xlim((0, 1))
plt.ylim((0, 1))
endfig(figures, "coverage-vs-accuracy")

# %%

# Accuracy vs. Length

lengths = numpy.asarray([len(i) for i in remapped_fasta.read_text().splitlines()[1::2]])
lengths = lengths[is_possible]
length_sorting = numpy.argsort(lengths)

# Surprise name change
if "ProtBert-BFD" in is_correct_per_method:
    is_correct_per_method["ProtBert BFD"] = is_correct_per_method["ProtBert-BFD"]

window_size = 1000
length_plot_data = {
    best_name: is_correct_per_method[best_tag][length_sorting],
    "ESM": is_correct_per_method["ESM"][length_sorting],
    "MMseqs2": is_correct_mmseqs[length_sorting],
    "SeqVec": is_correct_per_method["SeqVec LSTM1"][length_sorting],
    "ProtBert-BFD": is_correct_per_method["ProtBert BFD"][length_sorting],
}
numpy.savez(figures.joinpath("length-vs-accuracy.npz"), length_plot_data)

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
for label, is_correct_now in length_plot_data.items():
    rolling_mean_values = rolling_mean(is_correct_now, window_size)
    rolling_mean_lengths = rolling_mean(lengths[length_sorting], window_size)
    plt.plot(rolling_mean_lengths, rolling_mean_values, label=label)
plt.ylim((0, 1))
plt.xlabel(f"Rolling mean length over {window_size} entries")
plt.ylabel(f"Rolling mean accuracy over {window_size} entries")
plt.legend(loc="lower right")
plt.grid()
endfig(figures, "length-vs-accuracy")

# %%

length_plot_data = {
    best_name: is_correct_per_method[best_tag],
    "ESM": is_correct_per_method["ESM"],
    "MMseqs2": is_correct_mmseqs,
    "SeqVec": is_correct_per_method["SeqVec LSTM1"],
    "ProtBert BFD": is_correct_per_method["ProtBert BFD"],
}

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
bin_size = 50
buckets = 6
for name, data in length_plot_data.items():
    y = []
    y_err = []
    for start in numpy.arange(0, bin_size * (buckets - 1), bin_size):
        y_data = data[(lengths >= start) & (lengths < start + bin_size)]
        y.append(y_data.mean())
        y_err.append(y_data.std() / numpy.sqrt(len(y_data)))
    y_data = data[(lengths >= bin_size * (buckets - 1))]
    y.append(y_data.mean())
    y_err.append(y_data.std() / numpy.sqrt(len(y_data)))

    x_ticks = [
        f"{i}-{i + bin_size}"
        for i in numpy.arange(0, bin_size * (buckets - 1), bin_size)
    ]
    x_ticks.append(f">{bin_size * (buckets - 1)}")
    plt.errorbar(x=x_ticks, y=y, yerr=y_err, label=name)
plt.legend()
plt.ylim((0, 1))
plt.grid(axis="y")
plt.xlabel("Length bin")
plt.ylabel("QrawTop1")
plt.tight_layout()
endfig(figures, "length-vs-accuracy-binned")


# %%


def hist_evenly(sorted_x: ndarray, sorted_y: ndarray, bins: int, label: str):
    y = []
    y_err = []
    x_ticks = []
    for i in numpy.arange(0, bins):
        start = len(sorted_x) * i // (bins + 1)
        stop = len(sorted_x) * (i + 1) // (bins + 1)
        y_data = sorted_y[start:stop]
        y.append(y_data.mean())
        y_err.append(y_data.std() / numpy.sqrt(len(y_data)))

        x_ticks.append(f"{sorted_x[start]}-{sorted_x[stop]}")
    plt.errorbar(x=x_ticks, y=y, yerr=y_err, label=label)


length_plot_data = {
    best_name: is_correct_per_method[best_tag],
    "ESM": is_correct_per_method["ESM"],
    "MMseqs2": is_correct_mmseqs,
    "SeqVec": is_correct_per_method["SeqVec LSTM1"],
    "ProtBert BFD": is_correct_per_method["ProtBert BFD"],
}

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
bins = 7
for name, data in length_plot_data.items():
    hist_evenly(lengths[length_sorting], data[length_sorting], bins, name)
plt.legend()
plt.ylim((0, 1))
plt.grid(axis="y")
plt.xlabel(f"Length bin (1/{bins} of sequences per bin)")
plt.ylabel("QrawTop1")
plt.tight_layout()
endfig(figures, "length-vs-accuracy-binned2")

# %%

"""
Workflow for redundancy reducing CATH:
wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/sequence-data/cath-domain-seqs-S100.fa
cd dbs
mmseqs createdb ../cath-domain-seqs-S100.fa cath_100
mmseqs cluster --min-seq-id 0.2 --cluster-mode 3 cath_100 cath_clustered_3 /tmp
mmseqs createseqfiledb cath_100 cath_clustered_3 cath_clustered_3_seq
mmseqs result2flat cath_100 cath_100 cath_clustered_3_seq cath_clustered_3_seq.fasta
cat cath_clustered_3_seq.fasta | uniq > ../cath_clustered.fasta
"""

# %%

score_sorting = numpy.argsort(scores_best)
window_size = 1000
rolling_mean_values = rolling_mean(is_correct_best[score_sorting], window_size)
rolling_mean_scores = rolling_mean(scores_best[score_sorting], window_size)
plt.xlabel(f"Mean Cosine similarity in a rolling window of {window_size}")
plt.ylabel(f"Mean accuracy in a rolling window of {window_size}")
plt.plot(rolling_mean_scores, rolling_mean_values)
plt.ylim((0, 1))
plt.grid()
plt.show()

# %%

score_sorting = numpy.argsort(-scores_best)

plt.xlabel("Cosine similarity threshold")
plt.ylabel("QrawTop1 of hits over this cutoff")
thresholded_accuracy = is_correct_best[score_sorting].cumsum() / numpy.arange(
    1, len(score_sorting) + 1
)
plt.plot(scores_best[score_sorting], thresholded_accuracy)
plt.ylim((0, 1))
plt.grid()
plt.show()

# %%

plt.scatter(scores_best, e_values_cath, s=0.1)
plt.yscale("log")
plt.ylim(top=10**3, bottom=10 ** (-7))
plt.ylabel("E-value")
plt.xlabel("Cosine similarity")
plt.grid()
plt.tight_layout()
endfig(figures, "e_value_vs_cosine_scatter")

logged = numpy.log(e_values_cath)
logged[numpy.isinf(logged)] = -(10**9)  # doesn't really matter
print(pearsonr(scores_best, logged))
print(spearmanr(scores_best, e_values_cath))

# %%

plt.scatter(scores_best, e_values_cath, s=0.1)
plt.yscale("log")
plt.ylabel("E-value")
plt.xlabel("Cosine similarity")
plt.grid()
plt.tight_layout()
plt.show()

# %%
