#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import faiss
import numpy

from seqvec_search.data import LoadedData
from seqvec_search.main import faiss_search
from seqvec_search.tp_cumulative import figure_tp_cumulative, compute_tps_comulative
from seqvec_search.utils import configure_logging


def main():
    configure_logging()
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/pfam-dist"))
    # noinspection PyTypeChecker
    parser.add_argument(
        "index", type=Path, nargs="?", default=Path("data/pfam-dist/train-lsh-1024")
    )
    args = parser.parse_args()

    data = LoadedData.from_options(args.dataset, hits=500, knn_index=args.index)
    queries = numpy.load(str(data.test))
    # Search index with 500 hits
    results_lsh, _, _ = faiss_search(
        faiss.read_index(str(args.index)), queries, data.hits
    )
    tps_cumulative_lsh = compute_tps_comulative(data, results_lsh)
    # Search no index with 500 htis
    results_cosine, _, _ = faiss_search(numpy.load(str(data.train)), queries, data.hits)
    tps_cumulative_cosine = compute_tps_comulative(data, results_cosine)
    figure_tp_cumulative(
        ["Cosine", "LSH"],
        [tps_cumulative_cosine, tps_cumulative_lsh],
        "tp_cumulative_lsh",
    )


if __name__ == "__main__":
    main()
