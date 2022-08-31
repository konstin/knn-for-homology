import argparse
import json
from pathlib import Path
from typing import Set, Dict

import numpy

from seqvec_search.data import LoadedData
from seqvec_search.utils import configure_logging


def make_fasta_subset(
    input: Path, output: Path, ids_to_family: Dict[str, str], families: Set[str]
):
    with input.open() as in_fasta, output.open("w") as out_fasta:
        write = False
        for line in in_fasta:
            if line.startswith(">"):
                write = ids_to_family[line.strip()[1:]] in families
            if write:
                out_fasta.write(line)


def main():
    # Setup
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Benchmark approximate nearest neighbor search against MMSeqs2. See [TODO Paper Reference] "
    )
    # noinspection PyTypeChecker
    parser.add_argument("input", type=Path)
    # noinspection PyTypeChecker
    parser.add_argument("output", type=Path)
    parser.add_argument("families", type=int)
    args = parser.parse_args()

    args.output.mkdir()
    data = LoadedData.from_options(args.input)

    families = set(list(set(data.ids_to_family.values()))[: args.families])

    test_ids = list(filter(lambda x: data.ids_to_family[x] in families, data.test_ids))
    train_ids = list(
        filter(lambda x: data.ids_to_family[x] in families, data.train_ids)
    )

    args.output.joinpath("test.json").write_text(json.dumps(test_ids))
    args.output.joinpath("train.json").write_text(json.dumps(train_ids))

    test_filter = [data.ids_to_family[x] in families for x in data.test_ids]
    train_filter = [data.ids_to_family[x] in families for x in data.train_ids]

    numpy.save(
        args.output.joinpath("test.npy"), numpy.load(str(data.test))[test_filter]
    )
    numpy.save(
        args.output.joinpath("train.npy"), numpy.load(str(data.train))[train_filter]
    )

    args.output.joinpath("ids_to_family.json").write_text(
        json.dumps(data.ids_to_family)
    )
    make_fasta_subset(
        args.input.joinpath("train.fasta"),
        args.output.joinpath("train.fasta"),
        data.ids_to_family,
        families,
    )
    make_fasta_subset(
        args.input.joinpath("test.fasta"),
        args.output.joinpath("test.fasta"),
        data.ids_to_family,
        families,
    )


if __name__ == "__main__":
    main()
