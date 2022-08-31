"""
Script for creating a two small subsets of Pfam, one to be used as haystack and one for querying

Workflow notes
 In1: Pfam-A
 In2: pfamseq
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TextIO, Generator, Tuple, Dict, List

from tqdm import tqdm

from seqvec_search.utils import configure_logging


def fasta_generator(fp: TextIO) -> Generator[Tuple[str, str], None, None]:
    sequence = []
    header = None
    for line in fp:
        if line[0] == ">":
            if header:
                yield header, "".join(sequence)
            header = line.strip()[1:]
            sequence.clear()
        else:
            sequence.append(line.strip())
    yield header, "".join(sequence)


def make_pfam_subset(
    data: Path, seed: int, pfam_a: Path, pfamseq: Path, min: int, max: int
):
    picked_sequence = set()
    domain_extract_test: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    domain_extract_train: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    data.mkdir(exist_ok=True)
    picked_families = 0
    id_to_family = dict()
    rng = random.Random(seed)
    with pfam_a.open() as fp, data.joinpath("train.fasta").open(
        "w"
    ) as out_train, data.joinpath("test.fasta").open("w") as out_test:
        pbar = tqdm(total=18259)  # That's the value for pfam 33.1
        last_family = None
        entries = []
        for header, sequence in fasta_generator(fp):
            # >A0A1I4YJU4_9ENTR/160-195 A0A1I4YJU4.1 PF10417.10;1-cysPrx_C;
            last_space = header.rfind(" ")
            pfam_family = header[last_space + 1 : header.find(".", last_space)]
            if pfam_family != last_family:
                if len(entries) > min + max:
                    picked_families += 1
                    selected = rng.sample(entries, min + max)
                    split_size = rng.randint(min, max)
                    # > The resulting list is in selection order so that all sub-slices will also
                    # > be valid random samples.
                    for protein_id, domain_range, sequence_ in selected[:split_size]:
                        out_train.write(f">{protein_id}/{domain_range}\n{sequence_}\n")
                        domain_extract_train[protein_id][
                            protein_id + "/" + domain_range
                        ] = [domain_range]
                    for protein_id, domain_range, sequence_ in selected[split_size:]:
                        out_test.write(f">{protein_id}/{domain_range}\n{sequence_}\n")
                        domain_extract_test[protein_id][
                            protein_id + "/" + domain_range
                        ] = [domain_range]
                    for protein_id, domain_range, _ in selected:
                        picked_sequence.add(protein_id)
                        id_to_family[protein_id + "/" + domain_range] = last_family
                entries = []
                last_family = pfam_family
                pbar.update()
            protein_id, domain_range = header[: header.find(" ")].split("/")
            entries.append((protein_id, domain_range, sequence))
    print(f"Picked {picked_families} families")
    # Write annotations for full sequences
    with data.joinpath("extract_test.json").open("w") as fp:
        json.dump(domain_extract_test, fp)
    with data.joinpath("extract_train.json").open("w") as fp:
        json.dump(domain_extract_train, fp)
    with data.joinpath("ids_to_family.json").open("w") as fp:
        json.dump(id_to_family, fp)
    # Write full sequence
    with pfamseq.open() as pfamseq, data.joinpath("full-sequences.fasta").open(
        "w"
    ) as out:
        # The total is for pfamseq 33.1
        for header, sequence in tqdm(fasta_generator(pfamseq), total=50000000):
            sequence_id = header.split(" ")[1]
            if sequence_id in picked_sequence:
                picked_sequence.remove(sequence_id)
                out.write(f">{sequence_id}\n{sequence}\n")
    assert len(picked_sequence) == 0, picked_sequence


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("--pfam", type=Path, default="data")
    # noinspection PyTypeChecker
    parser.add_argument("--data", type=Path, default="data/pfam-dist")
    parser.add_argument("--min", type=int, default=7)
    parser.add_argument("--max", type=int, default=13)
    parser.add_argument(
        "--seed", type=int, default=532741831, help="The seed for the random generator"
    )
    args = parser.parse_args()

    make_pfam_subset(
        args.data,
        args.seed,
        args.pfam.joinpath("Pfam-A.fasta"),
        args.pfam.joinpath("pfamseq"),
        args.min,
        args.max,
    )


if __name__ == "__main__":
    main()
