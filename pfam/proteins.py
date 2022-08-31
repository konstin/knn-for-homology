# %%
import subprocess
import time
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy
from numpy import ndarray
from scipy.stats import sem
from tqdm import tqdm

from pfam.pfam_shared import build_domain_ranges, subset10, project_root, mmseqs_bin
from pfam.proteins_shared import (
    full_sequences_data,
    mmseqs_search,
    get_homologous_proteins,
    get_protein_to_domain,
    compute_auc1,
    full_sequences_fasta,
    db_dir,
)
from seqvec_search import mmseqs
from seqvec_search.mmseqs import write_prefilter_db, make_id_map
from seqvec_search.utils import endfig, E_VALUE_CUTOFF

figures = project_root.joinpath("more_sensitive/proteins-figures")

# Evaluate only a subset to improve performance
# NB: The plots in the manuscript were generated with slice(None, None, 1),
# but this uses more than 16GB memory
subsampler = slice(None, None, 1)
index = "hnsw"
# Activate this to compare with --max-seqs 600 for the claim that our
# performance is not due to doubling the number of sequences in the benchmark
test_600 = False
# Test mmseqs with --num-iterations 2 and -s 6 which is both faster and more sensitive than our method :/
test_iterated = False

smoothness = 300  # Higher means better plot but slower
limit = 300  # Make methods comparable

print(index, subsampler, test_600)

# %%

test = build_domain_ranges(subset10.joinpath("test.fasta"))
train = build_domain_ranges(subset10.joinpath("train.fasta"))
proteins = set(test) | set(train)
del test
del train
protein_to_domain = get_protein_to_domain(proteins)
homologous_proteins = get_homologous_proteins(protein_to_domain)
lines = full_sequences_fasta.read_text().splitlines(keepends=False)
original_full_sequences_ids = [line.split(" ")[1] for line in lines[::2]]
# original_full_sequences_ids = [line.split(" ")[0][1:] for line in lines[::2]]
full_sequences_ids = original_full_sequences_ids[subsampler]
full_sequences_lengths = numpy.asarray([len(line) for line in lines[1::2]])[
    subsampler
].copy()
del lines
protein_name_to_pos = {
    name: pos for pos, name in enumerate(original_full_sequences_ids)
}
# So PF07391 is a weird family. It's some small repeat and the only representative in Pfam-A is Q8I6U6_PLAF7,
# which has this domain 570 (!) times (http://pfam.xfam.org/family/NPR#tabview=tab1)
# This is a bogus action, but it prevent the division by zero and doesn't change the scores
# TODO: Remove this in the dataset creation
if "Q8I6U6_PLAF7" in homologous_proteins:
    homologous_proteins["Q8I6U6_PLAF7"].add("Q8I6U6_PLAF7")

homologous_proteins_int = [
    [protein_name_to_pos[i] for i in homologous_proteins[full_sequences_ids[index]]]
    for index in tqdm(range(len(full_sequences_ids)))
]
del protein_name_to_pos


# %%

# Technically, we're looking only at 299 hits; It would be better to redo this with --max-seqs 301
# and proper cutoff for k-nn and merged stuff


def remove_self_hit(hits: ndarray, scores: ndarray) -> Tuple[ndarray, ndarray]:
    """With lossy search it sometimes fails to put the self hit in front, so
    we use a little trickery to remove the self hit anyway"""
    second_hit_is_self_hit = numpy.argwhere(
        hits[:, 0] != numpy.arange(0, len(original_full_sequences_ids))[subsampler]
    )
    print(f"Fixing {len(second_hit_is_self_hit)} misplaced self hits")

    bogus = 0
    for missing_id in second_hit_is_self_hit[:, 0]:
        self_hit_id = numpy.arange(0, len(original_full_sequences_ids))[subsampler][
            missing_id
        ]
        if self_hit_id in hits[missing_id]:
            index = list(hits[missing_id]).index(self_hit_id)
        else:
            # Self hit not found at all? Just remove the last hit instead
            index = len(hits[missing_id]) - 1
            # print(f"Missing self hit for {missing_id}")
            bogus += 1
        hits[missing_id, 0], hits[missing_id, 1 : index + 1] = (
            hits[missing_id, index].copy(),
            hits[missing_id, 0:index].copy(),
        )
        scores[missing_id, 0], scores[missing_id, 1 : index + 1] = (
            scores[missing_id, index].copy(),
            scores[missing_id, 0:index].copy(),
        )

    print(f"There are {bogus} missing self hits")
    assert (
        numpy.sum(
            hits[:, 0] != numpy.arange(0, len(original_full_sequences_ids))[subsampler]
        )
        == bogus
    )

    return hits[:, 1:], scores[:, 1:]


# %%

if not full_sequences_data.joinpath(f"full_sequences_{index}_hits.npy").is_file():
    raise RuntimeError("Please run proteins_search.py")

t5_hits = numpy.load(full_sequences_data.joinpath(f"full_sequences_{index}_hits.npy"))[
    subsampler
].copy()
t5_scores = numpy.load(
    full_sequences_data.joinpath(f"full_sequences_{index}_scores.npy")
)[subsampler].copy()

t5_hits, t5_scores = remove_self_hit(t5_hits, t5_scores)

if not full_sequences_data.joinpath("full_sequences_mmseqs_hits.npy").is_file():
    # Prefilter: 12m 2s
    # Align: 5m 37s
    mmseqs_hits, mmseqs_e_values = mmseqs_search(
        full_sequences_fasta,
        full_sequences_data.joinpath("full_sequences_hits"),
        db_dir.joinpath("full_sequences_hits_db"),
        "full_sequences_mmseqs",
    )
else:
    mmseqs_hits = numpy.load(
        full_sequences_data.joinpath("full_sequences_mmseqs_hits.npy")
    )
    mmseqs_e_values = numpy.load(
        full_sequences_data.joinpath("full_sequences_mmseqs_e_values.npy")
    )

# from pathlib import Path
# db = Path("/home/schuetze/uniref30_expandaln/pfam_full_sequences")
# out = Path("/home/schuetze/uniref30_expandaln/res_ps_iter_2_s_6_pfam")
# fasta_ids = [i[1:].split(" ")[0] for i in full_sequences_fasta.read_text().splitlines()[::2]]
# mmseqs_hits, mmseqs_e_values = mmseqs.read_result_db_with_e_value(
#    fasta_ids, db, fasta_ids, db, out
# )
# mmseqs_hits, mmseqs_e_values = mmseqs.results_to_array(mmseqs_hits, mmseqs_e_values)
#
mmseqs_hits, mmseqs_e_values = remove_self_hit(
    mmseqs_hits[subsampler].copy(), mmseqs_e_values[subsampler].copy()
)

if test_iterated:
    if not full_sequences_data.joinpath(
        "full_sequences_mmseqs_iterated_hits.npy"
    ).is_file():
        mmseqs_iterated_hits, mmseqs_iterated_e_values = mmseqs_search(
            full_sequences_fasta,
            full_sequences_data.joinpath("full_sequences_hits"),
            db_dir.joinpath("full_sequences_hits_db"),
            "full_sequences_mmseqs_iterated_hits",
        )
    else:
        mmseqs_iterated_hits = numpy.load(
            full_sequences_data.joinpath("full_sequences_mmseqs_iterated_hits.npy")
        )
        mmseqs_iterated_e_values = numpy.load(
            full_sequences_data.joinpath("full_sequences_mmseqs_iterated_e_values.npy")
        )

    mmseqs_iterated_hits, mmseqs_iterated_e_values = remove_self_hit(
        mmseqs_iterated_hits[subsampler].copy(),
        mmseqs_iterated_e_values[subsampler].copy(),
    )

# %%

correct_totals = []
for index in tqdm(range(len(t5_hits))):
    all_correct = set(homologous_proteins_int[index])
    correct_totals.append(len(all_correct))
correct_totals = numpy.asarray(correct_totals)


def compute_correctness_array(full: ndarray) -> ndarray:
    correct = []
    for index, hits in enumerate(tqdm(full)):
        all_correct = set(homologous_proteins_int[index])
        correctness = [hit in all_correct for hit in hits]
        correct.append(correctness)
    return numpy.asarray(correct)


# %%

# Build an array with the combined hits
threshold = 0.1
combined_hits = []
combined_scores = []
for index, (
    t5_hits_row,
    t5_row_scores,
    mmseqs_hits_row,
    mmseqs_row_e_values,
) in enumerate(zip(tqdm(t5_hits), t5_scores, mmseqs_hits, mmseqs_e_values)):
    mmseqs_hits_set = set(mmseqs_hits_row[mmseqs_row_e_values < threshold])
    combined_hit_list = list(mmseqs_hits_row[mmseqs_row_e_values < threshold])
    non_zero_log = 10**-200  # It complains about zero log otherwise
    combined_hit_scores_list = list(
        -numpy.log(mmseqs_row_e_values[mmseqs_row_e_values < threshold] + non_zero_log)
    )
    for hit, score in zip(t5_hits_row, t5_row_scores):
        if len(combined_hit_list) == mmseqs_hits.shape[1]:
            break
        if hit not in mmseqs_hits_set:
            combined_hit_list.append(hit)
            combined_hit_scores_list.append(score)
    else:
        raise ValueError()
    assert len(combined_hit_list) == len(combined_hit_scores_list)
    combined_hits.append(combined_hit_list)
    combined_scores.append(combined_hit_scores_list)
combined_hits = numpy.asarray(combined_hits)
combined_scores = numpy.asarray(combined_scores)

# %%

# Align k-nn hits

max_hits = 300
knn_aligned_hits_npy = full_sequences_data.joinpath("knn_aligned_hits.npy")
knn_aligned_e_values_npy = full_sequences_data.joinpath("knn_aligned_e_values.npy")

if not knn_aligned_hits_npy.is_file():
    # Take k-nn hits, write to mmseqs prefilter database
    full_sequences_db = db_dir.joinpath("full_sequences_db")
    if not full_sequences_db.with_suffix(".dbtype").is_file():
        full_sequences_reindexed = full_sequences_fasta.with_name(
            "full_sequences_reindexed.fasta"
        )
        with full_sequences_fasta.open() as in_fasta, full_sequences_reindexed.open(
            "w"
        ) as out_fasta:
            for line in in_fasta:
                if line.startswith(">"):
                    sequence_id = line.split(" ")[1]
                    out_fasta.write(">" + sequence_id + "\n")
                else:
                    out_fasta.write(line)
        mmseqs.create_db(full_sequences_reindexed, full_sequences_db)

    knn_prefilter_db = db_dir.joinpath("full_sequences_knn_prefilter")
    if not knn_prefilter_db.with_suffix(".dbtype").is_file():
        full_sequence_id_map = make_id_map(
            original_full_sequences_ids, full_sequences_db
        )

        # We limit to MMseqs' max_hits for a fairer comparison
        # Use the full map but be careful with memory
        t5_full_mmap = numpy.load(
            full_sequences_data.joinpath("full_sequences_lsh_hits.npy"), mmap_mode="r"
        )
        write_prefilter_db(
            t5_full_mmap[:, :max_hits],
            knn_prefilter_db,
            numpy.arange(len(original_full_sequences_ids)),
            t5_full_mmap[:, :max_hits],
            full_sequence_id_map,
            full_sequence_id_map,
            clip=False,
        )
        del t5_full_mmap

    # Run mmseqs aligned
    if not db_dir.joinpath("full_sequences_knn_aligned.dbtype").is_file():
        start = time.time()
        subprocess.check_call(
            [
                str(mmseqs_bin),
                "align",
                "-e",
                str(E_VALUE_CUTOFF),
                str(full_sequences_db),
                str(full_sequences_db),
                knn_prefilter_db,
                str(db_dir.joinpath("full_sequences_knn_aligned")),
            ]
        )
        stop = time.time()
        db_dir.joinpath("full_sequences_knn_aligned.time.txt").write_text(
            str(stop - start)
        )

    # Read mmseqs k-nn aligned
    t5_aligned_hits, t5_aligned_e_values = mmseqs.read_result_db_with_e_value(
        original_full_sequences_ids,
        full_sequences_db,
        original_full_sequences_ids,
        full_sequences_db,
        db_dir.joinpath("full_sequences_knn_aligned"),
    )
    t5_aligned_hits, t5_aligned_e_values = mmseqs.results_to_array(
        t5_aligned_hits, t5_aligned_e_values
    )
    numpy.save(knn_aligned_hits_npy, t5_aligned_hits)
    numpy.save(knn_aligned_e_values_npy, t5_aligned_e_values)
else:
    t5_aligned_hits = numpy.load(knn_aligned_hits_npy)[subsampler].copy()
    t5_aligned_e_values = numpy.load(knn_aligned_e_values_npy)[subsampler].copy()
t5_aligned_hits = t5_aligned_hits[:, 1:]
t5_aligned_e_values = t5_aligned_e_values[:, 1:]

# %%

# combine ProtT5 and MMSeqs2 aligned

# Sort both together by e values
assert t5_aligned_hits.shape == mmseqs_hits.shape
both_aligned_hits_base = numpy.concatenate([mmseqs_hits, t5_aligned_hits], axis=1)
both_aligned_e_values_base = numpy.concatenate(
    [mmseqs_e_values, t5_aligned_e_values], axis=1
)

both_aligned_hits = []
both_aligned_e_values = []
for hits_row, scores_row in zip(
    tqdm(both_aligned_hits_base), both_aligned_e_values_base
):
    sorted_scores_row = numpy.argsort(scores_row)
    seen = set()
    both_aligned_hits_row = []
    both_aligned_e_values_row = []
    for hit, score in zip(hits_row[sorted_scores_row], scores_row[sorted_scores_row]):
        if hit in seen:
            continue
        both_aligned_hits_row.append(hit)
        both_aligned_e_values_row.append(score)
        seen.add(hit)
    sentinel_e_value = 10**6
    both_aligned_hits_row = both_aligned_hits_row[: mmseqs_hits.shape[1]]
    both_aligned_hits.append(
        numpy.pad(
            both_aligned_hits_row,
            (0, mmseqs_hits.shape[1] - len(both_aligned_hits_row)),
        )
    )
    both_aligned_e_values_row = both_aligned_e_values_row[: mmseqs_hits.shape[1]]
    both_aligned_e_values.append(
        numpy.pad(
            both_aligned_e_values_row,
            (0, mmseqs_hits.shape[1] - len(both_aligned_e_values_row)),
            constant_values=sentinel_e_value,
        )
    )
both_aligned_hits = numpy.asarray(both_aligned_hits)
both_aligned_e_values = numpy.asarray(both_aligned_e_values)

# %%

t5_auc1s = compute_auc1(
    t5_hits, homologous_proteins, full_sequences_ids, original_full_sequences_ids
)
mmseqs_auc1s = compute_auc1(
    mmseqs_hits, homologous_proteins, full_sequences_ids, original_full_sequences_ids
)
if test_iterated:
    # noinspection PyUnboundLocalVariable
    mmseqs_iterated_auc1s = compute_auc1(
        mmseqs_iterated_hits,
        homologous_proteins,
        full_sequences_ids,
        original_full_sequences_ids,
    )
t5_aligned_auc1s = compute_auc1(
    t5_aligned_hits,
    homologous_proteins,
    full_sequences_ids,
    original_full_sequences_ids,
)
both_aligned_auc1s = compute_auc1(
    both_aligned_hits,
    homologous_proteins,
    full_sequences_ids,
    original_full_sequences_ids,
)

combined_auc1s = compute_auc1(
    combined_hits, homologous_proteins, full_sequences_ids, original_full_sequences_ids
)

auc1s_all = {
    "MMseqs2 + knnProtT5 aligned": both_aligned_auc1s,
    f"MMseqs2 E<{threshold} + knnProtT5": combined_auc1s,
    "MMseqs2": mmseqs_auc1s,
    "knnProtT5 aligned": t5_aligned_auc1s,
    "knnProtT5": t5_auc1s,
}
if test_iterated:
    # noinspection PyUnboundLocalVariable
    auc1s_all["MMseqs2 iterated"] = mmseqs_iterated_auc1s

auc1s_plot = {
    "MMseqs2 + knnProtT5 aligned": both_aligned_auc1s,
    "MMseqs2": mmseqs_auc1s,
    "knnProtT5 aligned": t5_aligned_auc1s,
    "knnProtT5": t5_auc1s,
}

# %%

t5_correct = compute_correctness_array(t5_hits)
mmseqs_correct = compute_correctness_array(mmseqs_hits)
combined_correct = compute_correctness_array(combined_hits)
t5_aligned_correct = compute_correctness_array(t5_aligned_hits)
both_aligned_correct = compute_correctness_array(both_aligned_hits)

correct_all = {
    "MMseqs2 + knnProtT5 aligned": both_aligned_correct,
    f"MMseqs2 E<{threshold} + knnProtT5": combined_correct,
    "MMseqs2": mmseqs_correct,
    "knnProtT5 aligned": t5_aligned_correct,
    "knnProtT5": t5_correct,
}

if test_600:
    if not full_sequences_data.joinpath("full_sequences_mmseqs_hits.npy").is_file():
        mmseqs_600_hits, mmseqs_600_e_values = mmseqs_search(
            full_sequences_fasta,
            full_sequences_data.joinpath("full_sequences_hits"),
            db_dir.joinpath("full_sequences_hits_db_600"),
            "full_sequences_mmseqs_600",
            600,
        )
    else:
        mmseqs_600_hits = numpy.load(
            full_sequences_data.joinpath("full_sequences_mmseqs_600_hits.npy")
        )[subsampler].copy()
        mmseqs_600_e_values = numpy.load(
            full_sequences_data.joinpath("full_sequences_mmseqs_600_e_values.npy")
        )[subsampler].copy()
    mmseqs_600_hits, mmseqs_600_e_values = remove_self_hit(
        mmseqs_600_hits, mmseqs_600_e_values
    )
    mmseqs_600_auc1s = compute_auc1(
        mmseqs_600_hits,
        homologous_proteins,
        full_sequences_ids,
        original_full_sequences_ids,
    )
    auc1s_all["MMseqs2 600"] = mmseqs_600_auc1s
    auc1s_all["MMseqs2 600"] = mmseqs_600_auc1s
    mmseqs_600_correct = compute_correctness_array(mmseqs_600_hits)
    correct_all["MMseqs2 600"] = mmseqs_600_correct

# %%

for name, auc1s in auc1s_all.items():
    print(f"{name:<25} {auc1s.mean():.3f}±{sem(auc1s):.4f}")

print("Mean recall per query considering the first 300 hits")
for name, correct in correct_all.items():
    recall_per_query = correct[:, :300].sum(axis=1) / correct_totals
    print(
        f"{name:<25} {recall_per_query.mean():.1%}±{sem(recall_per_query.flatten()):.2%}"
    )

loss_aligning = (
    correct_all["knnProtT5"][:, :300].sum(axis=1)
    - correct_all["knnProtT5 aligned"][:, :300].sum(axis=1)
) / correct_totals

print(
    f"Loss through aligning: {loss_aligning.mean():.1%}±{sem(loss_aligning.flatten()):.2%}"
)

fraction_lost = (
    correct_all["knnProtT5"][:, :300].sum()
    - correct_all["knnProtT5 aligned"][:, :300].sum()
) / correct_totals.sum()
print(f"Fraction of hits lost aligning: {fraction_lost:.1%}")


# %%


def make_accuracy_over_hit(correct: ndarray) -> ndarray:
    # Normalize the count of correct by the total number of correct so that we get
    # the relative recall per query for each number of hits
    return (
        correct.cumsum(axis=1) / numpy.tile(correct_totals, (correct.shape[1], 1)).T
    ).mean(axis=0)


plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
for label, correct in correct_all.items():
    plt.plot(make_accuracy_over_hit(correct[:, :300]), label=label)
plt.ylim((0, 1))
plt.xlabel("Number of hits")
plt.ylabel("Mean fraction of homologous sequences found")
plt.grid()
plt.legend()
plt.tight_layout()
endfig(figures, f"accuracy_over_hits")

# %%

styles = [("C1", "dashdot"), ("C0", "solid"), ("C2", "dotted"), ("C3", "solid")]
plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
for (name, auc1s), (color, line_style) in zip(auc1s_plot.items(), styles):
    plt.plot(
        numpy.linspace(0, 1, len(auc1s)),
        auc1s[numpy.argsort(-auc1s)],
        label=f"{name} (mean: {auc1s.mean():.3f})",
        color=color,
        linestyle=line_style,
    )
plt.xlabel("AUC1 sensitivity")
plt.ylabel("Fraction of queries with at least this AUC1")
plt.grid()
plt.legend(loc="lower center")
plt.tight_layout()
endfig(figures, f"auc1")

# %%

sorted_lengths = numpy.argsort(-full_sequences_lengths)
median_length = full_sequences_lengths[sorted_lengths][len(full_sequences_lengths) // 2]
cumsum_meanifier = numpy.arange(1, len(full_sequences_lengths) + 1)

styles = [("C1", "dashdot"), ("C0", "solid"), ("C2", "dotted"), ("C3", "solid")]
plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
for ((label, auc1s), (color, line_style)) in zip(auc1s_plot.items(), styles):
    plt.plot(
        full_sequences_lengths[sorted_lengths],
        auc1s[sorted_lengths].cumsum() / cumsum_meanifier,
        label=label,
        color=color,
        linestyle=line_style,
    )
plt.vlines([median_length], 0, 1, color="black", label="Median protein length")
plt.xlim((0, 1000))
plt.ylim((0.2, 0.6))
plt.xlabel("Protein length (lower limit)")
plt.ylabel("AUC1")
plt.grid()
plt.legend()
plt.tight_layout()
endfig(figures, "protein_length_vs_auc1")

# %%

styles = [("C1", "dashdot"), ("C0", "solid"), ("C2", "dotted"), ("C3", "solid")]
plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
for ((label, auc1s), (color, line_style)) in zip(auc1s_plot.items(), styles):
    if "0.1" in label:
        continue
    limits = [200, 400, 600, 800, 1000]
    x_ticks = [f"<{limits[0]}"]
    bucketed = [auc1s[full_sequences_lengths < limits[0]]]
    # One is start, the next entry is stop
    for start, stop in zip(limits[:-1], limits[1:]):
        bucketed.append(
            auc1s[(full_sequences_lengths >= start) & (full_sequences_lengths < stop)]
        )
        x_ticks.append(f"{start}-{stop}")
    x_ticks.append(f">{limits[-1]}")
    bucketed.append(auc1s[full_sequences_lengths >= limits[-1]])

    plt.errorbar(
        x=x_ticks,
        y=[i.mean() for i in bucketed],
        yerr=[i.std() / numpy.sqrt(len(i)) for i in bucketed],
        label=label,
        color=color,
        linestyle=line_style,
    )

plt.xlabel("Length bucket")
plt.ylabel("AUC1 sensitivity")
plt.grid()
# plt.ylim((0, 1))
plt.legend()
plt.tight_layout()
endfig(figures, "length_bucketed_auc1")

# %%

plot_data = dict()

# precision-recall curve
if not figures.joinpath("precision_recall_curve.h5").is_file() or subsampler == slice(
    None, None, 1
):
    prec_recall_data = [
        (
            both_aligned_correct,
            -both_aligned_e_values,
            "MMseqs2 + knnProtT5 aligned (E-value)",
        ),
        (combined_correct, combined_scores, f"MMseqs2 E<{threshold} + knnProtT5"),
        (mmseqs_correct, -mmseqs_e_values, "MMseqs2 (E-value)"),
        (t5_aligned_correct, -t5_aligned_e_values, "knnProtT5 aligned (E-value)"),
        (t5_correct, t5_scores, "knnProtT5 (cosine similarity)"),
    ]
    if test_600:
        # noinspection PyUnboundLocalVariable
        prec_recall_data.append(
            (mmseqs_600_correct, -mmseqs_600_e_values, "MMseqs2 600 (E-value)")
        )
    for correct, scores, label in prec_recall_data:
        precision = []
        recall = []
        thresholds = numpy.quantile(
            scores[:, :limit], numpy.linspace(0, 1, smoothness + 1)
        )
        for score_threshold in tqdm(thresholds):
            # The precision is computed as the mean of the precision per query
            precision_inner = []
            recall_inner = []
            for correct_query, scores_query, total in zip(
                correct[:, :limit], scores[:, :limit], correct_totals
            ):
                selected_correct_query = correct_query[scores_query > score_threshold]
                tp = selected_correct_query.sum()
                if len(selected_correct_query) == 0:  # No prediction no error
                    precision_inner.append(1)
                else:
                    precision_inner.append(tp / len(selected_correct_query))
                recall_inner.append(tp / total)
            precision.append(numpy.mean(precision_inner))
            recall.append(numpy.mean(recall_inner))
        plot_data[label] = (recall, precision, thresholds)

    with h5py.File(figures.joinpath("precision_recall_curve.h5"), "w") as f:
        for label, (recall, precision, thresholds) in plot_data.items():
            f.create_dataset(f"{label}/recall", data=recall)
            f.create_dataset(f"{label}/precision", data=precision)
            f.create_dataset(f"{label}/thresholds", data=thresholds)
else:
    plot_data = dict()
    with h5py.File(figures.joinpath("precision_recall_curve.h5")) as f:
        for key, group in f.items():
            plot_data[key] = (
                group["recall"][:],
                group["precision"][:],
                group["thresholds"][:],
            )

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
style_map = {
    "MMseqs2 (E-value)": ("solid", "C0"),
    "MMseqs2 + knnProtT5 aligned (E-value)": ("dashdot", "C1"),
    "MMseqs2 E<0.1 + knnProtT5": ("--", "C5"),
    "knnProtT5 (cosine similarity)": ("-", "C3"),
    "knnProtT5 aligned (E-value)": ("dashed", "C2"),
}
for (label, (line_style, color)) in style_map.items():
    # renaming dirty fixup
    (recall, precision, _) = plot_data.get(label) or plot_data[label.replace("knnProtT5", "ProtT5 k-nn")]
    plt.plot(recall, precision, linestyle=line_style, color=color, label=label)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.grid()
plt.legend(loc="lower left")
plt.tight_layout()
endfig(figures, "precision_recall_curve")

# %%


plot_subset_scores = t5_scores[:, :limit]
plot_subset_correct = t5_correct[:, :limit]

score_buckets = numpy.linspace(0, 1 - (1 / smoothness), smoothness)

t5_quantized_precision = []
t5_quantized_precision_sem = []
for i in tqdm(score_buckets):
    bucket_mask = (i < plot_subset_scores) & (
        plot_subset_scores <= i + (1 / smoothness)
    )
    if bucket_mask.sum() == 0:
        continue
    else:
        bucket_correctness = plot_subset_correct[bucket_mask]
    t5_quantized_precision.append(bucket_correctness.mean())
    t5_quantized_precision_sem.append(sem(bucket_correctness))
t5_quantized_precision = numpy.asarray(t5_quantized_precision)
t5_quantized_precision_sem = numpy.asarray(t5_quantized_precision_sem)

plt.figure(figsize=(5, 5 * (4.8 / 6.4)))
# 1/(smoothness*2) is for centering
plt.scatter(
    score_buckets[-len(t5_quantized_precision) :] + (1 / (smoothness * 2)),
    t5_quantized_precision,
    s=0.5,
    label="ProtT5 knn",
)
plt.errorbar(
    score_buckets[-len(t5_quantized_precision) :] + (1 / (smoothness * 2)),
    t5_quantized_precision,
    yerr=t5_quantized_precision_sem,
    linestyle="none",
)
plt.xlabel(f"cosine similarity bucket (1/{smoothness})")
plt.ylabel("Accuracy")
# plt.ylim((0, 1))
plt.legend()
plt.grid()
plt.tight_layout()
endfig(figures, "cosine_bucketed_accuracy")

# %%
