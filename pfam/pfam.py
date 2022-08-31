# %%

# UniProt Release 2020_05 UniRef
import json
import subprocess
import time
from collections import Counter
from tempfile import TemporaryDirectory
from typing import List, Tuple, Iterable, Set, Dict, Optional

import matplotlib.pyplot as plt
import numpy
from numpy import ndarray
from tqdm import tqdm

from pfam.pfam_shared import (
    project_root,
    pfam,
    subset10,
    load_files,
    subset10_t5,
    mmseqs_bin,
)
from seqvec_search import mmseqs
from seqvec_search.mmseqs import write_prefilter_db, make_id_map
from seqvec_search.utils import endfig, rolling_mean, E_VALUE_CUTOFF

figures = project_root.joinpath("more_sensitive/pfam-figures")

dbs = pfam.joinpath("dbs")
dbs.mkdir(exist_ok=True)

train_db = dbs.joinpath("train")
test_db = dbs.joinpath("test")
out_db = dbs.joinpath("out")
prefilter_db = dbs.joinpath("prefilter")
prefilter_db_tsv = prefilter_db.with_suffix(".tsv")

load_files()

# %%

# Read results from search.py
train_ids = json.loads(subset10_t5.joinpath("train.json").read_text())
test_ids = json.loads(subset10_t5.joinpath("test.json").read_text())
ids_to_family = json.loads(subset10.joinpath("ids_to_family.json").read_text())
train_to_family = numpy.asarray([ids_to_family[i] for i in train_ids])
test_to_family = numpy.asarray([ids_to_family[i] for i in test_ids])
knn_scores = numpy.load(subset10_t5.joinpath("index_scores.npy"))
knn_hits = numpy.load(subset10_t5.joinpath("index_hits.npy"))

# %%

# Search with mmseqs

if not train_db.with_suffix(".dbtype").is_file():
    mmseqs.create_db(subset10.joinpath("train.fasta"), train_db)
if not test_db.with_suffix(".dbtype").is_file():
    mmseqs.create_db(subset10.joinpath("test.fasta"), test_db)

if not out_db.with_suffix(".dbtype").is_file():
    with TemporaryDirectory() as temp_dir:
        start = time.time()
        subprocess.check_call(
            [
                str(mmseqs_bin),
                "search",
                "-e",
                str(E_VALUE_CUTOFF),
                "-s",
                str(7.5),
                str(test_db),
                str(train_db),
                str(out_db),
                temp_dir,
            ]
        )
        end = time.time()
        print(f"MMseqs2 took {end - start}s")
        out_db.with_suffix(".time.txt").write_text(str(end - start))

# 41427147 total hits
# 1039s on X1

mmseqs_hits, mmseqs_e_values = mmseqs.read_result_db_with_e_value(
    train_ids, train_db, test_ids, test_db, out_db
)

# %%

# Search with mmseqs iterated

out_iterated_db = dbs.joinpath("out_iterated")
if not out_iterated_db.with_suffix(".dbtype").is_file():
    with TemporaryDirectory() as temp_dir:
        start = time.time()
        subprocess.check_call(
            [
                str(mmseqs_bin),
                "search",
                "-e",
                str(E_VALUE_CUTOFF),
                "-s",
                str(7.5),
                str(test_db),
                str(train_db),
                str(out_iterated_db),
                temp_dir,
                "--num-iterations",
                str(3),
            ]
        )
        end = time.time()
        print(f"MMseqs2 iterated took {end - start}s")
        out_iterated_db.with_suffix(".time.txt").write_text(str(end - start))

# 41427147 total hits
# lsf-server-7: 325.82653427124023s

mmseqs_iterated_hits, mmseqs_iterated_e_values = mmseqs.read_result_db_with_e_value(
    train_ids, train_db, test_ids, test_db, out_iterated_db
)

# %%

# noinspection PyTypeChecker
mmseqs_correct: Dict[int, ndarray] = {}
for query, hits in tqdm(mmseqs_hits.items(), total=len(mmseqs_hits)):
    # noinspection PyTypeChecker
    mmseqs_correct[query] = test_to_family[query] == train_to_family[hits]

# %%

knn_correct = []
for i in range(len(knn_hits)):
    # noinspection PyTypeChecker
    correctness: ndarray = train_to_family[knn_hits[i]] == test_to_family[i]
    knn_correct.append(correctness)
knn_correct = numpy.asarray(knn_correct)

# %%

max_hits = max(len(hits) for hits in mmseqs_hits.values())
mmseqs_correct_array = []
mmseqs_e_value_array = []
for i in range(len(mmseqs_correct)):
    mmseqs_correct_array.append(
        numpy.pad(mmseqs_correct[i], (0, max_hits - len(mmseqs_correct[i])))
    )
    # With E<10000, 100000 is bigger than all others
    mmseqs_e_value_array.append(
        numpy.pad(
            mmseqs_e_values[i],
            (0, max_hits - len(mmseqs_e_values[i])),
            constant_values=100000,
        )
    )
mmseqs_correct_array = numpy.asarray(mmseqs_correct_array)
mmseqs_e_value_array = numpy.asarray(mmseqs_e_value_array)
top_hit_correct = mmseqs_correct_array[:, 0]
top_hit_e_value = mmseqs_e_value_array[:, 0]
print(f"Top hit correct MMseqs2 s=8: {numpy.mean(top_hit_correct.mean()):.1%}")

# %%

e_value_argsort = numpy.argsort(top_hit_e_value)

# For each possible cutoff, take MMseqs2 for values below the cutoff
# and knn for values above it
combined = (
    numpy.cumsum(top_hit_correct[e_value_argsort])
    + numpy.cumsum(knn_correct[:, 0][e_value_argsort][::-1])[::-1]
)
combined_accuracy = combined / len(combined)
mmseqs_with_cutoff_accuracy = numpy.cumsum(top_hit_correct[e_value_argsort]) / len(
    combined
)
e_value_sorted = top_hit_e_value[e_value_argsort]

plt.axhline(top_hit_correct.mean(), color="black", label="MMSeqs2 baseline")
plt.axhline(knn_correct[:, 0].mean(), color="green", label="k-nn")
plt.plot(e_value_sorted, mmseqs_with_cutoff_accuracy, label="MMseqs2 with cutoff")
plt.plot(e_value_sorted, combined_accuracy, label="MMSeqs2 E<1 + k-nn")
plt.xlim((10**-12, 10**6))
plt.ylim((0, 1))
plt.grid()
plt.xlabel("E-value cutoff")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()
numpy.savez(
    figures.joinpath("combining-mmseqs-and-knn.npz"),
    e_value_sorted=e_value_sorted,
    combined_accuracy=combined_accuracy,
    mmseqs_with_cutoff_accuracy=mmseqs_with_cutoff_accuracy,
    top_hit_correct=top_hit_correct.mean(),
    knn_1_correct=knn_correct[:, 0].mean(),
)
endfig(figures, "combining-mmseqs-and-knn")

# %%

e_cutoff = 1
combined_e1 = top_hit_correct.copy()
combined_e1[top_hit_e_value >= e_cutoff] = knn_correct[:, 0][
    top_hit_e_value >= e_cutoff
]
combined_e1_scores = -top_hit_e_value.copy()
combined_e1_scores[top_hit_e_value >= e_cutoff] = -knn_scores[
    top_hit_e_value >= e_cutoff, 0
]

plt.plot(
    numpy.linspace(0, 1, len(top_hit_correct)),
    (
        numpy.cumsum(top_hit_correct[numpy.argsort(top_hit_e_value)])
        / numpy.arange(1, len(top_hit_correct) + 1)
    ),
    label="MMseqs2",
)
plt.plot(
    numpy.linspace(0, 1, len(combined_e1)),
    (
        numpy.cumsum(combined_e1[numpy.argsort(-combined_e1_scores)])
        / numpy.arange(1, len(combined_e1) + 1)
    ),
    label="MMseqs2 E<1 + k-nn",
)
plt.plot(
    numpy.linspace(0, 1, len(knn_correct[:, 0])),
    (
        numpy.cumsum(knn_correct[:, 0][numpy.argsort(knn_scores[:, 0])])
        / numpy.arange(1, len(knn_correct[:, 0]) + 1)
    ),
    label="k-nn",
)
plt.xlabel("Fraction of annotated queries")
plt.ylabel("Accuracy of annotated queries")
plt.legend()
plt.grid()
endfig(figures, "coverage-vs-accuracy")


# %%


def accuracy_by_e_value(name: str, xlim: bool):
    top_hit_sorting = numpy.argsort(-top_hit_e_value)
    window_size = 1000

    e_values_rolling_mean = rolling_mean(top_hit_e_value[top_hit_sorting], window_size)
    rolling_mean_knn = rolling_mean(knn_correct[:, 0][top_hit_sorting], window_size)
    rolling_mean_mmseqs = rolling_mean(top_hit_correct[top_hit_sorting], window_size)

    plt.plot(e_values_rolling_mean, rolling_mean_mmseqs, label="MMseqs2")
    plt.plot(e_values_rolling_mean, rolling_mean_knn, label="k-nn")
    plt.xscale("log")
    plt.xlabel(f"Rolling mean E-value over {window_size} hits")
    plt.ylabel(f"Rolling mean accuracy over {window_size} hits")
    plt.ylim((-0.05, 1.05))
    if xlim:
        plt.xlim((10**-10, 10**3))
    plt.grid()
    plt.legend()
    numpy.savez(
        figures.joinpath(name),
        top_hit_e_value=top_hit_e_value,
        knn_1_correct=knn_correct[:, 0],
        top_hit_correct=top_hit_correct,
        top_hit_sorting=top_hit_sorting,
        window_size=window_size,
    )
    endfig(figures, name)


accuracy_by_e_value("accuracy-by-e-value", True)


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

        x_ticks.append(f"{(sorted_x[start]):0.0E} {(sorted_x[stop]):0.0E}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.1)
    plt.errorbar(x=x_ticks, y=y, yerr=y_err, label=label, marker="v")


def accuracy_by_e_value2():
    top_hit_sorting = numpy.argsort(-top_hit_e_value)

    length_plot_data = {"MMseqs2": top_hit_correct, "T5": knn_correct[:, 0]}

    bins = 10
    for name, data in length_plot_data.items():
        hist_evenly(top_hit_e_value[top_hit_sorting], data[top_hit_sorting], bins, name)
    plt.grid(axis="y")
    plt.xlabel(f"E-value bucket (1/{bins} of sequences per bucket)")
    plt.ylabel(f"Accuracy")
    plt.ylim((-0.05, 1.05))
    plt.legend()
    endfig(figures, "accuracy-by-e-value-binned")


accuracy_by_e_value2()

# %%

if not prefilter_db_tsv.is_file():
    if not prefilter_db.with_suffix(".dbtype").is_file():
        subprocess.check_call(
            [
                str(mmseqs_bin),
                "prefilter",
                "-s",
                str(7.5),
                "--max-seqs",
                str(300),
                str(test_db),
                str(train_db),
                str(prefilter_db),
            ]
        )

    subprocess.check_call(
        [
            str(mmseqs_bin),
            "createtsv",
            str(test_db),
            str(train_db),
            str(prefilter_db),
            str(prefilter_db_tsv),
        ]
    )

# %%

mmseqs_tp: List[Set[int]] = []
knn_tp: List[Set[int]] = []

for i in range(len(knn_hits)):
    correct_family = test_to_family[i]
    knn_tp.append(set(knn_hits[i][train_to_family[knn_hits[i]] == correct_family]))
    mmseqs_tp.append(set(mmseqs_hits[i][mmseqs_correct[i]]))

tp_count_mmseqs = 0
tp_count_both = 0
tp_count_knn = 0
for mmseqs_set, knn_set in zip(mmseqs_tp, knn_tp):
    tp_count_mmseqs += len(mmseqs_set - knn_set)
    tp_count_both += len(mmseqs_set & knn_set)
    tp_count_knn += len(knn_set - mmseqs_set)

print(
    "TP Set overlap",
    f"Only MMseqs2: {tp_count_mmseqs / (10 * len(knn_hits)):.1%}",
    f"Both: {tp_count_both / (10 * len(knn_hits)):.1%}",
    f"Only knn: {tp_count_knn / (10 * len(knn_hits)):.1%}",
)

# %%


combined_for_auc1 = {}
e_value_cutoff = 1
for key in mmseqs_hits.keys():
    from_mmseqs = mmseqs_hits[key][mmseqs_e_values[key] < e_value_cutoff]
    from_knn = knn_hits[key, :10]
    concatenated = numpy.concatenate((from_mmseqs, from_knn))
    (_, indices) = numpy.unique(concatenated, return_index=True)
    combined_for_auc1[key] = concatenated[indices[numpy.argsort(indices)]]

# noinspection PyTypeChecker
combined_correct: Dict[int, ndarray] = {}
for query, hits in tqdm(combined_for_auc1.items(), total=len(combined_for_auc1)):
    # noinspection PyTypeChecker
    combined_correct[query] = test_to_family[query] == train_to_family[hits]

combined_correct_array = []
knn_aligned_max_hits = max(len(hits) for hits in combined_for_auc1.values())
for i in range(len(combined_for_auc1)):
    combined_correct_array.append(
        numpy.pad(
            combined_correct[i], (0, knn_aligned_max_hits - len(combined_for_auc1[i]))
        )
    )
combined_correct_array = numpy.asarray(combined_correct_array)


# %%


def evaluate(
    results: Iterable[Tuple[int, ndarray]], limit: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """Returns the AUC1 and TP values"""
    family_sizes_ = dict(Counter(train_to_family))
    auc1s = []
    tps = []
    for name, matches in results:
        correct_family_ = test_to_family[name]
        if limit:
            matches = matches[:limit]
        tp = numpy.sum(train_to_family[matches] == correct_family_)
        auc1_ = 0
        for i in matches:
            if train_to_family[i] == correct_family_:
                auc1_ += 1
            else:
                break
        auc1s.append(auc1_ / family_sizes_[correct_family_])
        tps.append(tp / family_sizes_[correct_family_])
    return auc1s, tps


number_of_hits = 10
# noinspection PyTypeChecker
auc1s_mmseqs, tps_mmseqs = evaluate(mmseqs_hits.items(), number_of_hits)
# noinspection PyTypeChecker
auc1s_mmseqs_iterated, tps_mmseqs_iterated = evaluate(
    mmseqs_iterated_hits.items(), number_of_hits
)
print(
    f"MMseqs2 mean AUC1: {numpy.mean(auc1s_mmseqs):.3f}, "
    f"Mean TP ({number_of_hits}): {numpy.mean(tps_mmseqs):.3f}"
)
print(
    f"MMseqs2 iterated mean AUC1: {numpy.mean(auc1s_mmseqs_iterated):.3f}, "
    f"Mean TP ({number_of_hits}): {numpy.mean(tps_mmseqs_iterated):.3f}"
)
# noinspection PyTypeChecker
auc1s_mmseqs_knn, tps_mmseqs_knn = evaluate(combined_for_auc1.items(), number_of_hits)
print(
    f"combined mean AUC1: {numpy.mean(auc1s_mmseqs_knn):.3f}, "
    f"Mean TP ({number_of_hits}): {numpy.mean(tps_mmseqs_knn):.3f}, "
    f"Mean TP ({max_hits}): {(combined_correct_array[:, :max_hits].sum(axis=1) / 10).mean():.3f}"
)
auc1_knn, tp_knn = evaluate(enumerate(knn_hits), number_of_hits)
print(
    f"k-nn mean AUC1: {numpy.mean(auc1_knn):.3f}, "
    f"Mean TP ({number_of_hits}): {numpy.mean(tp_knn):.3f}, "
    f"Mean TP ({max_hits}): {(knn_correct[:, :max_hits].sum(axis=1) / 10).mean():.3f}"
)

# MMseqs2 mean AUC1: 0.659, Mean TP (10): 0.686
# MMseqs2 iterated mean AUC1: 0.743, Mean TP (10): 0.769
# combined mean AUC1: 0.738, Mean TP (10): 0.802, Mean TP (300): 0.820
# k-nn mean AUC1: 0.565, Mean TP (10): 0.641, Mean TP (300): 0.839

# %%

test_faiss_to_mmseqs = make_id_map(test_ids, test_db)
train_faiss_to_mmseqs = make_id_map(train_ids, train_db)

# %%

if not dbs.joinpath("knn_prefilter.dbtype").is_file():
    # We limit to MMseqs' max_hits for a fairer comparison
    write_prefilter_db(
        knn_hits[:, :max_hits],
        dbs.joinpath("knn_prefilter"),
        numpy.arange(len(test_ids)),
        knn_scores[:, :max_hits],
        test_faiss_to_mmseqs,
        train_faiss_to_mmseqs,
        clip=False,
    )

if not dbs.joinpath("knn_aligned.dbtype").is_file():
    subprocess.check_call(
        [
            str(mmseqs_bin),
            "align",
            "-e",
            str(E_VALUE_CUTOFF),
            str(test_db),
            str(train_db),
            dbs.joinpath("knn_prefilter"),
            str(dbs.joinpath("knn_aligned")),
        ]
    )

# %%

knn_aligned_hits, knn_aligned_e_values = mmseqs.read_result_db_with_e_value(
    train_ids, train_db, test_ids, test_db, dbs.joinpath("knn_aligned")
)

knn_aligned_correct: Dict[int, ndarray] = {}
for query, hits in tqdm(knn_aligned_hits.items(), total=len(knn_aligned_hits)):
    # noinspection PyTypeChecker
    knn_aligned_correct[query] = test_to_family[query] == train_to_family[hits]

knn_aligned_correct_array = []
knn_aligned_max_hits = max(len(hits) for hits in knn_aligned_hits.values())
for i in range(len(knn_aligned_correct)):
    knn_aligned_correct_array.append(
        numpy.pad(
            knn_aligned_correct[i],
            (0, knn_aligned_max_hits - len(knn_aligned_correct[i])),
        )
    )
knn_aligned_correct_array = numpy.asarray(knn_aligned_correct_array)

knn_aligned_e_value_array = []
for i in range(len(knn_aligned_correct)):
    # With E<10000, 100000 is bigger than all others
    knn_aligned_e_value_array.append(
        numpy.pad(
            knn_aligned_e_values[i],
            (0, knn_aligned_correct_array.shape[1] - len(knn_aligned_e_values[i])),
            constant_values=100000,
        )
    )
knn_aligned_e_value_array = numpy.asarray(knn_aligned_e_value_array)

# noinspection PyTypeChecker
auc1s_knn_aligned, tps_knn_aligned = evaluate(knn_aligned_hits.items())
print(
    f"k-nn aligned mean AUC1: {numpy.mean(auc1s_knn_aligned):.2f}, "
    f"Mean TP (10) {(knn_aligned_correct_array[:, :10].sum(axis=1) / 10).mean():.2f}, "
    f"Mean TP ({max_hits}): {(knn_aligned_correct_array[:, :max_hits].sum(axis=1) / 10).mean():.2f}"
)

# k-nn aligned mean AUC1: 0.69, Mean TP (10) 0.73, Mean TP (300): 0.79

# %%

cumulative_tp_knn = knn_correct.mean(axis=0).cumsum() / 10
cumulative_tp_knn_aligned = knn_aligned_correct_array.mean(axis=0).cumsum() / 10
cumulative_tp_mmseqs = mmseqs_correct_array.mean(axis=0).cumsum() / 10
numpy.savez(
    figures.joinpath("tp.npz"),
    cumulative_tp_knn=cumulative_tp_knn,
    cumulative_tp_mmseqs=cumulative_tp_mmseqs,
)
plt.plot(cumulative_tp_knn, label="k-nn")
plt.plot(cumulative_tp_mmseqs, label="MMseqs2")
plt.plot(cumulative_tp_knn_aligned, label="k-nn + alignment")
plt.xlabel("Number of hits")
plt.ylabel("Fraction of TP")
plt.xlim((0, 300))
plt.ylim((0.6, 1))
plt.legend()
plt.grid()
endfig(figures, "tp")

# %%

for name, limit in [("first_10", slice(10)), ("300", slice(max_hits))]:
    total_to_be_found = sum(Counter(train_to_family).values()) * 10

    plot_data = []
    for scores, correct, label in [
        (mmseqs_e_value_array, mmseqs_correct_array, "mmseqs"),
        (knn_scores, knn_correct, "k-nn"),
        (knn_aligned_e_value_array, knn_aligned_correct_array, "k-nn + alignment"),
    ]:
        sorting = numpy.argsort(scores[:, limit].flatten())
        sorted_correct = correct[:, limit].flatten()[sorting]
        del sorting
        precision = numpy.cumsum(sorted_correct) / numpy.arange(
            1, len(sorted_correct) + 1
        )
        recall = numpy.cumsum(sorted_correct) / total_to_be_found

        # To fit in memory, we do some subsampling
        subsample = numpy.asarray(
            [numpy.searchsorted(recall, i) for i in numpy.linspace(0, 1, 10000)]
        ).clip(max=len(recall) - 1)

        plot_data.append((recall[subsample], precision[subsample], label))
    for recall, precision, label in plot_data:
        plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    numpy.savez(
        figures.joinpath(f"precision_recall_{name}"),
        recall=list(zip(*plot_data))[0],
        precision=list(zip(*plot_data))[1],
        label=list(zip(*plot_data))[2],
    )
    endfig(figures, f"precision_recall_{name}")

# %%

auc1s_optimal = []

family_sizes = dict(Counter(train_to_family))
for i in tqdm(range(len(knn_hits))):
    knn_hits_set = set()
    mmseqs_hits_set = set()
    correct_family = test_to_family[i]
    for j in knn_aligned_hits[i]:
        if train_to_family[j] == correct_family:
            knn_hits_set.add(j)
        else:
            break
    for j in mmseqs_hits[i]:
        if train_to_family[j] == correct_family:
            mmseqs_hits_set.add(j)
        else:
            break

    auc1s_optimal.append(
        len(knn_hits_set | mmseqs_hits_set) / family_sizes[correct_family]
    )

auc1s_optimal = numpy.asarray(auc1s_optimal)
print(f"MMseqs1 mean AUC1 {auc1s_optimal.mean():.2f}")

# %%

auc1s_joined = []
j_ = 0
k_ = 0
family_sizes = dict(Counter(train_to_family))
for i in tqdm(range(len(knn_hits))):
    auc1 = 0
    correct_family = test_to_family[i]
    j = 0
    k = 0
    picked = set()
    while True:
        if len(knn_aligned_e_values[i]) == j and len(mmseqs_e_values[i]) == k:
            break
        if len(mmseqs_e_values[i]) == k or (
            len(knn_aligned_e_values[i]) > j
            and knn_aligned_e_values[i][j] <= mmseqs_e_values[i][k]
        ):
            chosen = knn_aligned_hits[i][j]
            j += 1
        else:
            chosen = mmseqs_hits[i][k]
            k += 1

        # Don't count hits twice
        if chosen in picked:
            continue
        picked.add(chosen)

        if train_to_family[chosen] == correct_family:
            auc1 += 1
        else:
            break
    j_ += j
    k_ += k

    auc1s_joined.append(auc1 / family_sizes[correct_family])
print("?", j_, k_)
auc1s_joined = numpy.asarray(auc1s_joined)
print("? AUC1 joined", auc1s_joined.mean())

# %%

# plt.hist(
#    auc1s_optimal,
#    bins=sorted(set(auc1s_optimal)),
#    cumulative=-1,
#    histtype="step",
#    label="MMSeqs + k-nn theoretical limit",
# )
plt.hist(
    auc1s_joined,
    bins=sorted(set(auc1s_joined)),
    cumulative=-1,
    histtype="step",
    label="MMSeqs + k-nn aligned",
)
plt.hist(
    auc1s_mmseqs_knn,
    bins=sorted(set(auc1s_mmseqs_knn)),
    cumulative=-1,
    histtype="step",
    label="MMSeqs E<1 + k-nn",
)
plt.hist(
    auc1s_mmseqs,
    bins=sorted(set(auc1s_mmseqs)),
    cumulative=-1,
    histtype="step",
    label="MMSeqs2",
)
plt.hist(
    auc1_knn, bins=sorted(set(auc1_knn)), cumulative=-1, histtype="step", label="k-nn"
)
# plt.hist(
#    auc1s_knn_aligned,
#    bins=sorted(set(auc1s_knn_aligned)),
#    cumulative=-1,
#    histtype="step",
#    label="k-nn aligned",
# )
plt.xlabel("AUC1")
plt.ylabel("Number of queries reaching this AUC1")
plt.legend()
plt.grid()
endfig(figures, "auc1")

# %%
