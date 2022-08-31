#!/usr/bin/env python3
import argparse
import logging
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Union, Iterable

import faiss
import matplotlib.pyplot as plt
import numpy
from numpy import ndarray

from seqvec_search import mmseqs
from seqvec_search.constants import default_hits
from seqvec_search.data import LoadedData
from seqvec_search.utils import configure_logging

logger = logging.getLogger(__name__)


def faiss_search(
    haystack: Union[ndarray, faiss.IndexLSH],
    queries: ndarray,
    hits: int = default_hits,
    metric=faiss.METRIC_INNER_PRODUCT,
) -> Tuple[ndarray, ndarray, float]:
    """Searches the haystack for queries and returns the specified number of hits for each"""
    start = time.time()
    if metric == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(queries)
    if isinstance(haystack, ndarray):
        if metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(haystack)
        index = faiss.IndexFlat(haystack.shape[1], metric)
        # noinspection PyArgumentList
        index.train(haystack)
        # noinspection PyArgumentList
        index.add(haystack)
    else:
        index = haystack
    logging.info(f"Preprocessing took {time.time() - start}s")
    start = time.time()
    # noinspection PyArgumentList
    scores, result = index.search(queries, hits)
    search_time = time.time() - start
    logging.info(
        f"Searching {len(queries)} samples and {hits} hits took {search_time}s"
    )
    return result, scores, search_time


def evaluate_faiss(
    data: LoadedData, results: ndarray
) -> Tuple[List[float], List[float]]:
    """Converts faiss ids to string ids and calls evaluate"""
    generator = (
        (data.test_ids[key], [data.train_ids[i] for i in row])
        for key, row in enumerate(results)
    )
    return evaluate(data, generator)


def evaluate(
    data: LoadedData, results: Iterable[Tuple[str, Iterable[str]]]
) -> Tuple[List[float], List[float]]:
    """Returns the AUC1 and TP values"""
    family_sizes = dict(Counter(data.ids_to_family[i] for i in data.train_ids))
    auc1s = []
    tps = []
    for name, matches in results:
        correct_family = data.ids_to_family[name]
        tp = sum(data.ids_to_family[i] == correct_family for i in matches)
        auc1 = 0
        for i in matches:
            if data.ids_to_family[i] == correct_family:
                auc1 += 1
            else:
                break
        auc1s.append(auc1 / family_sizes[correct_family])
        tps.append(tp / family_sizes[correct_family])
    return auc1s, tps


def make_figure(
    figure_dir: Path,
    scores_list: List[List[float]],
    labels: List[str],
    score: str,
    filename: str,
    svg: bool = False,
):
    # Save the data to allow "reproducing" the cath-figures and to allow making stylistic changes without rerunning the code
    datafile = figure_dir.joinpath(filename.split(".")[0] + "-data.npz")
    np_data = {key: numpy.asarray(value) for key, value in zip(labels, scores_list)}
    numpy.savez(str(datafile), **np_data)

    # Actually plot
    for scores, label in zip(scores_list, labels):
        scores = numpy.flip(numpy.sort(numpy.asarray(scores)))
        plt.plot(numpy.linspace(0, 1, len(scores)), scores, label=label)
    plt.xlabel(
        f"Fraction of queries with at least this {score} (n={len(scores_list[0])})"
    )
    plt.ylabel(score)
    plt.legend()
    plt.grid()
    plt.savefig(str(figure_dir.joinpath(filename)))
    if svg:
        plt.savefig(str(figure_dir.joinpath(filename).with_suffix(".svg")))
    plt.close()


def main():
    # Setup
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Benchmark approximate nearest neighbor search against MMSeqs2. See [TODO Paper Reference] "
    )
    # noinspection PyTypeChecker
    parser.add_argument("dataset", type=Path)
    # noinspection PyTypeChecker
    parser.add_argument("--knn-index", type=Path)
    parser.add_argument("--hits", type=int, default=default_hits)
    args = parser.parse_args()

    data = LoadedData.from_options(args.dataset, args.hits, args.knn_index)
    queries = numpy.load(str(data.test))

    # k-NN
    if args.knn_index:
        knn_index = faiss.read_index(str(args.knn_index))
    else:
        knn_index = numpy.load(str(data.train))

    results, scores, search_time = faiss_search(knn_index, queries, data.hits)
    auc1s_knn, tps_knn = evaluate_faiss(data, results)
    logger.info(
        f"Mean AUC1 for k-NN: {numpy.mean(auc1s_knn):f}, "
        f"Mean TP: {numpy.mean(tps_knn):f}, "
        f"Time: {int(search_time)}s"
    )
    make_figure(args.dataset, [auc1s_knn], ["k-NN"], "AUC1", "auc1_knn.jpg")

    # k-NN + Alignment
    mmseqs.write_prefilter_db_data(
        data, numpy.arange(queries.shape[0]), results, scores
    )
    align_time = mmseqs.align(data)
    logger.info("Evaluating k-NN + Alignment")
    hits = mmseqs.read_result_db(data, data.mmseqs_dir.joinpath("result_combined"))

    # noinspection PyTypeChecker
    auc1s_knn_alignment, tps_knn_alignment = evaluate(data, hits.items())
    make_figure(
        args.dataset,
        [auc1s_knn_alignment],
        ["k-NN + Alignment"],
        "AUC1",
        "auc1_knn_alignment.jpg",
    )
    logger.info(
        f"Mean AUC1 for k-NN + Alignment: {numpy.mean(auc1s_knn_alignment):f}, "
        f"Mean TP: {numpy.mean(tps_knn_alignment):f}, "
        f"Time: {int(search_time + align_time)}s"
    )

    # MMseqs2
    mmseqs_time = mmseqs.search(data)
    logger.info("Evaluating MMseqs2")
    hits = mmseqs.read_result_db(data, data.mmseqs_dir.joinpath("result_mmseqs2"))

    # noinspection PyTypeChecker
    auc1s_mmseqs2, tps_mmseqs2 = evaluate(data, hits.items())
    make_figure(args.dataset, [auc1s_mmseqs2], ["MMseqs2"], "AUC1", "auc1_mmseqs2.jpg")
    logger.info(
        f"Mean AUC1 for MMseqs2: {numpy.mean(auc1s_mmseqs2):f}, Mean TP: {numpy.mean(tps_mmseqs2):f}, Time {int(mmseqs_time)}s"
    )

    # Summary plot
    make_figure(
        args.dataset,
        [auc1s_knn, auc1s_knn_alignment, auc1s_mmseqs2],
        ["k-NN", "k-NN + Alignment", "MMseqs2"],
        "AUC1",
        "auc1.jpg",
    )

    results = [
        ("k-NN", auc1s_knn, search_time),
        ("k-NN + Alignment", auc1s_knn_alignment, search_time + align_time),
        ("MMseqs2", auc1s_mmseqs2, mmseqs_time),
    ]
    print("name                 AUC1  SD    time")
    for name, auc1s, measured_time in results:
        print(
            f"{name:20} {numpy.mean(auc1s):.3f} {numpy.std(auc1s):.3f} {int(measured_time)}s"
        )


if __name__ == "__main__":
    main()
