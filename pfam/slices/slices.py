# %%
from collections import defaultdict
from itertools import groupby
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy
from numpy import ndarray
from tqdm import tqdm

from pfam.pfam_shared import build_domain_ranges, subset10
from pfam.proteins_shared import (
    full_sequences_data,
    mmseqs_search,
    get_homologous_proteins,
    get_protein_to_domain,
    compute_auc1,
    full_sequences_fasta,
)
from pfam.slices.slices_shared import (
    slices_fasta,
    slice_len,
    slices_data,
    slices_db_dir,
)

# %%

test = build_domain_ranges(subset10.joinpath("test.fasta"))
train = build_domain_ranges(subset10.joinpath("train.fasta"))
proteins = set(test) | set(train)

slices = []
for header in slices_fasta.read_text().splitlines()[::2]:
    domain, start = header[1:].split("-")
    slices.append((domain, (int(start), int(start) + slice_len)))

# %%


protein_to_domain = get_protein_to_domain(proteins)
homologous_proteins = get_homologous_proteins(protein_to_domain)
full_sequences_ids = [
    line.split(" ")[1] for line in full_sequences_fasta.read_text().splitlines()[::2]
]

# %%

slices_pfams_matching = []
slices_pfams_intersecting = []
for protein, (slice_start, slice_stop) in slices:
    matching = set()
    intersecting = set()
    for pfam, (domain_start, domain_stop) in protein_to_domain[protein]:
        #  |--------------slice---------------|
        #           |-----domain----------|
        if slice_start <= domain_start and domain_stop <= slice_stop:
            matching.add(pfam)
        #  |--------------slice---------------|
        #           |-----domain-------------------|
        # Or:

        #            |--------------slice---------------|
        #  |-----domain-------------------|
        elif slice_start < domain_stop and domain_start < slice_stop:
            intersecting.add(pfam)
    slices_pfams_matching.append(matching)
    slices_pfams_intersecting.append(intersecting)

family_sizes = defaultdict(int)
for family_set in slices_pfams_matching:
    for family in family_set:
        family_sizes[family] += 1
family_sizes = dict(family_sizes)

has_annotation = numpy.asarray([len(i) == 1 for i in slices_pfams_matching])
slices_pfams_matching_annotated = []
for i in slices_pfams_matching:
    if len(i) == 1:
        slices_pfams_matching_annotated.append(list(i)[0])
# %%

slices_hits = numpy.load(slices_data.joinpath("slices_hits.npy"), mmap_mode="r")[
    :, :100
].copy()
slices_scores = numpy.load(slices_data.joinpath("slices_scores.npy"), mmap_mode="r")[
    :, :100
].copy()

slices_mmseqs_hits = numpy.load(
    slices_data.joinpath("slices_mmseqs_hits.npy"), mmap_mode="r"
)[:, :100].copy()
slices_mmseqs_e_values = numpy.load(
    slices_data.joinpath("slices_mmseqs_e_values.npy"), mmap_mode="r"
)[:, :100].copy()


# %%


def evaluate(
    trimmed_down: ndarray, ignore_unannotated: bool = False
) -> Tuple[ndarray, ndarray, ndarray]:
    is_correct = []
    is_ignore = []
    for index, hits, annotation in zip(
        numpy.arange(len(slices_pfams_matching))[has_annotation],
        trimmed_down,
        tqdm(slices_pfams_matching_annotated),
    ):
        is_correct.append([annotation in slices_pfams_matching[hit] for hit in hits])
        is_ignore.append(
            [
                (
                    (annotation in slices_pfams_intersecting[hit])
                    or (ignore_unannotated and slices_pfams_matching[hit] == set())
                )
                for hit in hits
            ]
        )
    is_correct = numpy.asarray(is_correct)
    is_ignore = numpy.asarray(is_ignore)

    auc1s = []
    for index, family, correct, ignore in zip(
        numpy.arange(len(slices_pfams_matching))[has_annotation],
        slices_pfams_matching_annotated,
        tqdm(is_correct),
        is_ignore,
    ):
        auc1_ = 0
        for correct_hit, ignore_hit in zip(correct, ignore):
            if correct_hit:
                auc1_ += 1
            elif ignore_hit:
                continue
            else:
                break
        auc1s.append(auc1_ / family_sizes[family])
    auc1s = numpy.asarray(auc1s)

    return is_correct, is_ignore, auc1s


# %%

t5_is_correct, t5_is_ignore, t5_auc1s = evaluate(slices_hits[has_annotation])
mmseqs_is_correct, mmseqs_is_ignore, mmseqs_auc1s = evaluate(
    slices_mmseqs_hits[has_annotation]
)

_, _, t5_auc1s_ignore_unannotated = evaluate(
    slices_hits[has_annotation], ignore_unannotated=True
)
_, _, mmseqs_auc1s_ignore_unannotated = evaluate(
    slices_mmseqs_hits[has_annotation], ignore_unannotated=True
)

# %%

plt.plot(
    numpy.linspace(0, 1, len(t5_auc1s)),
    t5_auc1s[numpy.argsort(-t5_auc1s)],
    label="T5 k-nn",
)
plt.plot(
    numpy.linspace(0, 1, len(mmseqs_auc1s)),
    mmseqs_auc1s[numpy.argsort(-mmseqs_auc1s)],
    label="MMseqs2",
)
plt.plot(
    numpy.linspace(0, 1, len(t5_auc1s_ignore_unannotated)),
    t5_auc1s_ignore_unannotated[numpy.argsort(-t5_auc1s_ignore_unannotated)],
    label="T5 k-nn ignore_unannotated",
)
plt.plot(
    numpy.linspace(0, 1, len(mmseqs_auc1s_ignore_unannotated)),
    mmseqs_auc1s_ignore_unannotated[numpy.argsort(-mmseqs_auc1s_ignore_unannotated)],
    label="MMseqs2 ignore_unannotated",
)
plt.legend()
plt.grid()
plt.xlabel("AUC1")
plt.ylabel("Fraction of slices with at least this AUC1 (cumulative)")
plt.tight_layout()
plt.show()

# %%

slices_db_dir.mkdir(exist_ok=True)
slices_db = slices_db_dir.joinpath("slices")
slices_hits_db = slices_db_dir.joinpath("slices_hits_db")
full_sequences_db = slices_db_dir.joinpath("full_sequences")
full_sequences_hits_db = slices_db_dir.joinpath("full_sequences_hits_db")

# MMseqs took 980s
mmseqs_slices_hits, mmseqs_slices_e_values = mmseqs_search(
    slices_fasta, slices_db, slices_hits_db, "slices_mmseqs"
)
mmseqs_full_sequences_hits, mmseqs_full_sequences_e_values = mmseqs_search(
    full_sequences_fasta,
    full_sequences_db,
    full_sequences_hits_db,
    "full_sequences_mmseqs",
)

# %%

plt.plot(
    numpy.linspace(0, 1, len(t5_auc1s)), t5_auc1s[numpy.argsort(t5_auc1s)], label="T5"
)
plt.plot(
    numpy.linspace(0, 1, len(mmseqs_auc1s)),
    mmseqs_auc1s[numpy.argsort(mmseqs_auc1s)],
    label="MMseqs2",
)
plt.grid()
plt.ylabel("Mean AUC1")
plt.xlabel("Fraction of hits (cumulative)")
plt.title("SLICES: auc1 cumulative (sorted)")
plt.legend()
plt.show()

# %%

plt.plot(mmseqs_is_correct.cumsum(axis=1).sum(axis=0) / len(t5_is_correct), label="T5")
plt.plot(t5_is_correct.cumsum(axis=1).sum(axis=0) / len(t5_is_correct), label="MMseqs2")
plt.title("SLICES: Mean TP per hit over the number of hits")
plt.ylabel("TP found")
plt.xlabel("Number of hits")
plt.grid()
plt.legend()
plt.show()

# %%

t5_full = numpy.load(full_sequences_data.joinpath("full_sequences_hits.npy"))[:, :100]
mmseqs_full = numpy.load(full_sequences_data.joinpath("full_sequences_mmseqs_hits.npy"))
mmseqs_full_e_values = numpy.load(
    full_sequences_data.joinpath("full_sequences_mmseqs_e_values.npy")
)

# %%

full_t5_auc1s = compute_auc1(
    t5_full, homologous_proteins, full_sequences_ids, full_sequences_ids
)
full_mmseqs_auc1s = compute_auc1(
    mmseqs_full, homologous_proteins, full_sequences_ids, full_sequences_ids
)


# %%

# Assemble fragment hits
def assemble(
    slices_hits_: ndarray, slices_scores_: ndarray
) -> Tuple[ndarray, List[str]]:
    assembled_proteins_ = []
    assembled_is_correct_ = []
    for (protein, protein_slices) in groupby(
        enumerate(tqdm(slices)), lambda x: x[1][0]
    ):
        assembled_proteins_.append(protein)
        protein_slices = list(protein_slices)
        hits_set = []
        scores_set = []
        for index, _ in protein_slices:
            hits_set.append(slices_hits_[index])
            scores_set.append(slices_scores_[index])
        hits = numpy.asarray(hits_set).flatten()
        scores = numpy.asarray(scores_set).flatten()
        # Sort by scores and normalize length
        hits = hits[numpy.argsort(-scores)]

        is_correct = []
        all_correct = homologous_proteins[protein]
        picked = set()
        for hit in hits[: slices_hits_.shape[1]]:
            hit_protein = slices[hit][0]
            # ignore duplicate hits
            if hit_protein in picked:
                continue
            is_correct.append(hit_protein in all_correct)
            picked.add(hit_protein)
        if len(is_correct) < slices_hits_.shape[1]:
            is_correct = is_correct + (
                [False] * (slices_hits_.shape[1] - len(is_correct))
            )
        assembled_is_correct_.append(is_correct)
    return numpy.asarray(assembled_is_correct_), assembled_proteins_


def auc1_assembled(assembled_is_correct_: ndarray, assembled_proteins_: List[str]):
    assembled_auc1s = []
    for is_correct, protein in zip(assembled_is_correct_, assembled_proteins_):
        all_correct = homologous_proteins[protein]
        auc1 = 0
        for single_match in is_correct:
            if single_match:
                auc1 += 1
            else:
                break
        assembled_auc1s.append(auc1 / len(all_correct))
    return numpy.asarray(assembled_auc1s)


# %%

assembled_is_correct, assembled_proteins = assemble(slices_hits, slices_scores)
assembled_auc1s = auc1_assembled(assembled_is_correct, assembled_proteins)
assembled_mmseqs_is_correct, assembled_mmseqs_proteins = assemble(
    slices_mmseqs_hits, mmseqs_slices_e_values
)
assembled_mmseqs_auc1s = auc1_assembled(
    assembled_mmseqs_is_correct, assembled_mmseqs_proteins
)

# %%


slice_counts = []
for _, protein_slices in groupby(slices, lambda x: x[0]):
    slice_counts.append(len(list(protein_slices)))
slice_counts = numpy.asarray(slice_counts)

# %%
assert len(full_t5_auc1s) == len(full_mmseqs_auc1s)
assert len(full_t5_auc1s) == len(assembled_auc1s)

print(
    f"T5 full AUC1: {full_t5_auc1s.mean():.3f}\n"
    f"MMseqs full AUC1: {full_mmseqs_auc1s.mean():.3f}\n"
    f"T5 assembled AUC1: {assembled_auc1s.mean():.3f}\n"
    f"MMseqs assembled AUC1: {assembled_mmseqs_auc1s.mean():.3f}\n"
)

print(
    f"T5 full AUC1: {full_t5_auc1s[slice_counts > 1].mean():.3f}\n"
    f"MMseqs full AUC1: {full_mmseqs_auc1s[slice_counts > 1].mean():.3f}\n"
    f"T5 assembled AUC1: {assembled_auc1s[slice_counts > 1].mean():.3f}\n"
    f"MMseqs assembled AUC1: {assembled_mmseqs_auc1s[slice_counts > 1].mean():.3f}\n"
)

plt.plot(
    numpy.linspace(0, 1, len(full_mmseqs_auc1s)),
    full_mmseqs_auc1s[numpy.argsort(full_mmseqs_auc1s)],
    label=f"MMseqs2 full (mean: {full_mmseqs_auc1s.mean():.3f})",
)
plt.plot(
    numpy.linspace(0, 1, len(assembled_mmseqs_auc1s)),
    assembled_mmseqs_auc1s[numpy.argsort(assembled_mmseqs_auc1s)],
    label=f"MMseqs2 assembled (mean: {assembled_mmseqs_auc1s.mean():.3f})",
)
plt.plot(
    numpy.linspace(0, 1, len(assembled_auc1s)),
    assembled_auc1s[numpy.argsort(assembled_auc1s)],
    label=f"T5 assembled (mean: {assembled_auc1s.mean():.3f})",
)
plt.plot(
    numpy.linspace(0, 1, len(full_t5_auc1s)),
    full_t5_auc1s[numpy.argsort(full_t5_auc1s)],
    label=f"T5 full (mean: {full_t5_auc1s.mean():.3f})",
)
plt.grid()
plt.legend()
plt.title("Pfam full sequences auc1 cumulative (sorted)")
plt.show()
plt.close()

# %%
