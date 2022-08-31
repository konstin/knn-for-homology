"""
The 3 layers of the full embeddings of the pfam subset are to big, so we break them down here.

There are three kind of descriptors used in this script:
 * The sequence id, e.g. "Q7R7H1_PLAYO"
 * The domain id, e.g. "Q7R7H1_PLAYO/291-334"
 * The full annotation, e.g. "Q7R7H1_PLAYO/291-334 Q7R7H1.1 PF09689.10;PY_rept_46;"

The reason is that the pfamseq file contains sequences with sequence ids, which we want to map to
the corresponding pfam fasta files contains domains with domain ids. Finally we want to write out
the full annotation because it contains the family.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

import numpy
from numpy import ndarray

from pfam.pfam_shared import build_domain_ranges
from seqvec_search.utils import configure_logging

if TYPE_CHECKING:
    from seqvec.seqvec import EmbedderReturnType


def extract_domain_representations_from_embedder(
    embedder: "EmbedderReturnType", domain_ranges_test: dict, domain_ranges_train: dict
) -> Tuple[Dict[str, ndarray], Dict[str, ndarray]]:
    data_train = dict()
    data_test = dict()
    for sequence_id, embedding in embedder:
        # Start is one based (-1), stop is inclusive (+1) but also one based (-1)
        for (start, stop, annotation) in domain_ranges_train[sequence_id]:
            data_train[annotation] = embedding[start - 1 : stop].mean(axis=0)
        for (start, stop, annotation) in domain_ranges_test[sequence_id]:
            data_test[annotation] = embedding[start - 1 : stop].mean(axis=0)
    return data_test, data_train


def main():
    from seqvec.seqvec import get_embeddings

    configure_logging()
    parser = argparse.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("--data", type=Path, default="data/pfam-dist")
    # noinspection PyTypeChecker
    parser.add_argument("--model-dir", type=Path, default="../model")
    args = parser.parse_args()

    domain_ranges_train = build_domain_ranges(args.data.joinpath("train.fasta"))
    domain_ranges_test = build_domain_ranges(args.data.joinpath("test.fasta"))
    print(f"Train has {sum(len(i) for i in domain_ranges_train.values())} entries")
    print(f"Test has {sum(len(i) for i in domain_ranges_test.values())} entries")

    embedder = get_embeddings(
        args.data.joinpath("full-sequences.fasta"),
        model_dir=args.model_dir,
        id_field=0,
        batchsize=120000,
        layer="all",
    )
    data_test, data_train = extract_domain_representations_from_embedder(
        embedder, domain_ranges_test, domain_ranges_train
    )

    args.data.joinpath("train.json").write_text(json.dumps(list(data_train.keys())))
    train_full = numpy.asarray(list(data_train.values()))
    numpy.save(args.data.joinpath("train_full.npy"), train_full)
    args.data.joinpath("test.json").write_text(json.dumps(list(data_test.keys())))
    test_full = numpy.asarray(list(data_test.values()))
    numpy.save(args.data.joinpath("test_full.npy"), test_full)
    # Extract LSTM1
    numpy.save(args.data.joinpath("train.npy"), train_full[:, 1024:2048])
    numpy.save(args.data.joinpath("test.npy"), test_full[:, 1024:2048])


if __name__ == "__main__":
    main()
