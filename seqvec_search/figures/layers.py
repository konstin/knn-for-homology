#!/usr/bin/env python3
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Callable

import numpy
from numpy import ndarray

from seqvec_search import mmseqs
from seqvec_search.constants import figure_dir
from seqvec_search.data import LoadedData
from seqvec_search.main import faiss_search, evaluate_faiss, make_figure, evaluate
from seqvec_search.tp_cumulative import compute_tps_comulative, figure_tp_cumulative
from seqvec_search.utils import configure_logging

logger = logging.getLogger(__name__)

hits = 1000
align_hits = 500


def main():
    configure_logging()
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("dataset", type=Path, default=Path("data/pfam-dist"))
    args = parser.parse_args()
    pfam_dist: Path = args.dataset

    logger.info(f"Loading the data")
    train_full = numpy.load(str(pfam_dist.joinpath("train.npy")))
    test_full = numpy.load(str(pfam_dist.joinpath("test.npy")))
    data = LoadedData.from_options(pfam_dist, hits)

    operations: List[Tuple[str, Callable[[ndarray], ndarray]]] = [
        (f"LSTM 1", lambda x: x[:, 1024:2048]),
        (
            f"CNN + LSTM 1 + LSTM 2 (baseline)",
            lambda x: x[:, :1024] + x[:, 1024:2048] + x[:, 2048:],
        ),
        (f"CNN and LSTM 1 and LSTM 2", lambda x: x),
        (f"LSTM 2", lambda x: x[:, 2048:]),
        (f"LSTM 1 and LSTM 2", lambda x: x[:, 1024:]),
        (f"LSTM 1 + LSTM 2", lambda x: x[:, 1024:2048] + x[:, 2048:]),
        (f"LSTM 1 - LSTM 2", lambda x: x[:, 1024:2048] - x[:, 2048:]),
        (f"CNN", lambda x: x[:, :1024]),
    ]

    layer_names = []
    layer_auc1s = []
    layer_tp_cumulative = []
    layer_auc1s_knn_alignment = []
    for name, tranformation in operations:
        layer_names.append(name)

        # Make subvectors
        logger.info(f"Building array for {name}")
        train = numpy.ascontiguousarray(tranformation(train_full))
        test = numpy.ascontiguousarray(tranformation(test_full))

        # Search
        results, scores, search_time = faiss_search(train, test, hits)
        auc1s_knn, tps_knn = evaluate_faiss(data, results)
        layer_auc1s.append(auc1s_knn)

        layer_tp_cumulative.append(compute_tps_comulative(data, results))

        # k-NN + Alignment
        mmseqs.write_prefilter_db_data(
            data,
            numpy.arange(results.shape[0]),
            results[:, :align_hits],
            scores[:, :align_hits],
        )
        align_time = mmseqs.align(data)
        logger.info("Evaluating k-NN + Alignment")
        mmseqs_hits = mmseqs.read_result_db(
            data, data.mmseqs_dir.joinpath("result_combined")
        )
        # noinspection PyTypeChecker
        auc1s_knn_alignment, tps_knn_alignment = evaluate(data, mmseqs_hits.items())
        layer_auc1s_knn_alignment.append(auc1s_knn_alignment)
        logger.info(
            f"Mean AUC1 for k-NN + Alignment: {numpy.mean(auc1s_knn_alignment):f}, "
            f"Mean TP: {numpy.mean(tps_knn_alignment):f}, "
            f"Time: {int(search_time + align_time)}s"
        )

    figure_tp_cumulative(layer_names, layer_tp_cumulative, "layers_tp_cumulative")

    # AUC1 unaligned
    figure_dir.mkdir(exist_ok=True)
    make_figure(
        figure_dir,
        layer_auc1s,
        layer_names,
        "AUC1",
        "layers_auc1_unaligned.jpg",
        svg=True,
    )

    # AUC1 aligned
    make_figure(
        figure_dir,
        layer_auc1s_knn_alignment,
        layer_names,
        "AUC1",
        "layers_auc1.jpg",
        svg=True,
    )


if __name__ == "__main__":
    main()
