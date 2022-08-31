import re
from collections import defaultdict
from itertools import chain
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from time import time
from typing import Tuple, Dict, List, Set

import numpy
import pandas
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from pfam.pfam_shared import pfam, pfam_a, mmseqs_bin
from seqvec_search import mmseqs

full_sequences_data = pfam.joinpath("full_sequences_data")
db_dir = full_sequences_data.joinpath("dbs")
full_sequences_fasta = full_sequences_data.joinpath("full_sequences.fasta")


def mmseqs_search(
    fasta: Path,
    db: Path,
    out: Path,
    npy_prefix: str,
    max_seqs: int = 300,
    s: float = 7.5,
    mmseqs_command_extra=None,
) -> Tuple[ndarray, ndarray]:
    mmseqs_hits_npy = full_sequences_data.joinpath(f"{npy_prefix}_hits.npy")
    if mmseqs_hits_npy.is_file():
        mmseqs_hits = numpy.load(mmseqs_hits_npy)
        mmseqs_e_values = numpy.load(
            full_sequences_data.joinpath(f"{npy_prefix}_e_values.npy")
        )
        return mmseqs_hits, mmseqs_e_values
    print(f"MMseqs search for {npy_prefix}")
    with TemporaryDirectory() as temp_dir:
        check_call([str(mmseqs_bin), "createdb", str(fasta), str(db)])

        start = time()
        mmseqs_command = [
            str(mmseqs_bin),
            "search",
            "-s",
            str(s),
            "-e",
            str(10000),
            "--max-seqs",
            str(max_seqs),
            str(db),
            str(db),
            str(out),
            temp_dir,
        ]
        check_call(mmseqs_command + (mmseqs_command_extra or []))
        stop = time()
        print(f"MMseqs2 took {stop - start}")
        out.with_suffix(".time.txt").write_text(str(stop - start))

    fasta_ids = [i[1:].split(" ")[0] for i in fasta.read_text().splitlines()[::2]]
    mmseqs_hits, mmseqs_e_values = mmseqs.read_result_db_with_e_value(
        fasta_ids, db, fasta_ids, db, out
    )
    mmseqs_hits, mmseqs_e_values = mmseqs.results_to_array(mmseqs_hits, mmseqs_e_values)
    numpy.save(mmseqs_hits_npy, mmseqs_hits)
    numpy.save(
        full_sequences_data.joinpath(f"{npy_prefix}_e_values.npy"), mmseqs_e_values
    )
    return mmseqs_hits, mmseqs_e_values


def get_homologous_proteins(
    protein_to_domain: Dict[str, List[str]]
) -> Dict[str, Set[str]]:
    protein_domains = {
        protein: set(i[0] for i in domains)
        for protein, domains in protein_to_domain.items()
    }

    domain_proteins = defaultdict(set)
    for protein, domains in protein_domains.items():
        for domain in domains:
            domain_proteins[domain].add(protein)
    domain_proteins = dict(domain_proteins)

    # Note: This object is huge for some reason
    homologous_proteins = dict()
    for protein, domains in protein_domains.items():
        homologs = set(chain(*(domain_proteins[domain] for domain in domains)))
        homologs.remove(protein)
        homologous_proteins[protein] = homologs

    return homologous_proteins


def get_protein_to_domain(proteins: Set[str]) -> Dict[str, List[str]]:
    pfam_a_cache_h5 = full_sequences_data.joinpath("pfam_a_cache.h5")
    if pfam_a_cache_h5.is_file():
        # noinspection PyTypeChecker
        pfam_a_cache: DataFrame = pandas.read_hdf(pfam_a_cache_h5, key="pfam_a_cache")
        protein_to_domain = defaultdict(list)
        for (protein, family, start, stop) in pfam_a_cache.itertuples(index=False):
            protein_to_domain[protein].append((family, (start, stop)))
    else:
        # >A0A1I4YJU4_9ENTR/160-195 A0A1I4YJU4.1 PF10417.10;1-cysPrx_C;
        #  ^^^^^^^^^^^^^^^^ ^^^ ^^^              ^^^^^^^
        header_re = re.compile(r">(.+)/(\d+)-(\d+) .* (.*)\.\d+;.*;")
        protein_to_domain = defaultdict(list)
        with pfam_a.open() as fp:
            for line in tqdm(fp):
                if line[0] != ">":
                    continue
                protein, start, stop, family = header_re.match(line).groups()
                if protein not in proteins:
                    continue
                # Inclusive 1-based indexing
                start = int(start) - 1
                stop = int(stop)
                protein_to_domain[protein].append((family, (start, stop)))

        records = []
        for protein, value in protein_to_domain.items():
            for (family, (start, stop)) in value:
                records.append((protein, family, start, stop))

        pfam_a_cache = DataFrame.from_records(
            records, columns=["protein", "family", "start", "stop"]
        )
        pfam_a_cache.to_hdf(pfam_a_cache_h5, key="pfam_a_cache")

    del pfam_a_cache
    return protein_to_domain


def compute_auc1(
    hits: ndarray,
    homologous_proteins: Dict[str, Set[str]],
    queries: List[str],
    target_ids: List[str],
) -> ndarray:
    auc1s = []
    for index, hits in enumerate(hits):
        # if len(protein_domains[full_sequences_ids[index]]) == 1:
        #    continue
        all_correct = homologous_proteins[queries[index]]
        auc1 = 0
        for hit in hits:
            if target_ids[hit] in all_correct:
                auc1 += 1
            else:
                break
        auc1s.append(auc1 / max(len(all_correct), 1))
    return numpy.asarray(auc1s)
