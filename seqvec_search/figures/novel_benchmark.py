#!/usr/bin/env python3
import logging
from argparse import ArgumentParser
from pathlib import Path

import faiss
import numpy
import pandas

from seqvec_search import mmseqs
from seqvec_search.constants import figure_dir
from seqvec_search.data import LoadedData
from seqvec_search.main import faiss_search, evaluate, make_figure
from seqvec_search.utils import configure_logging

logger = logging.getLogger(__name__)


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

    data = LoadedData.from_options(args.dataset, knn_index=args.index)
    queries = numpy.load(str(data.test))
    knn_index = faiss.read_index(str(args.index))

    hit_counts = [2000, 1500, 1000, 500, 250, 100, 50]
    auc1s_knn_alignments = []
    tps_knn_alignments = []
    knn_times = []
    align_times = []
    for hits in hit_counts:
        # k-NN
        results, scores, knn_time = faiss_search(knn_index, queries, hits)
        knn_times.append(knn_time)

        # k-NN + Alignment
        mmseqs.write_prefilter_db_data(
            data, numpy.arange(queries.shape[0]), results, scores
        )
        align_times.append(mmseqs.align(data))
        mmseqs_hits = mmseqs.read_result_db(
            data, data.mmseqs_dir.joinpath("result_combined")
        )
        # noinspection PyTypeChecker
        auc1s_knn_alignment, tps_knn_alignment = evaluate(data, mmseqs_hits.items())
        auc1s_knn_alignments.append(auc1s_knn_alignment)
        tps_knn_alignments.append(tps_knn_alignment)
        logger.info(
            f"Mean AUC1 for k-NN + Alignment: {numpy.mean(auc1s_knn_alignment):f}"
        )

    make_figure(
        figure_dir,
        auc1s_knn_alignments,
        [str(i) for i in hit_counts],
        "AUC1",
        "lsh_hits",
        svg=True,
    )

    auc1s_knn_alignment_means = [
        numpy.mean(auc1s_knn_alignment) for auc1s_knn_alignment in auc1s_knn_alignments
    ]
    tps_knn_alignment_means = [
        numpy.mean(tps_knn_alignment) for tps_knn_alignment in tps_knn_alignments
    ]
    total_times = [a + b for a, b in zip(knn_times, align_times)]
    rows = zip(
        auc1s_knn_alignment_means,
        tps_knn_alignment_means,
        total_times,
        knn_times,
        align_times,
    )
    df = pandas.DataFrame(
        rows,
        columns=["AUC1", "TP", "Total", "Prefiltering", "Alignment"],
        index=hit_counts,
    )
    df.index.name = "Hits"
    Path("data/cath-figures/novel_benchmark.csv").write_text(df.to_csv())
    markdown = df.to_markdown(floatfmt=".3f")
    print(markdown)
    Path("data/cath-figures/novel_benchmark.md").write_text(markdown)


if __name__ == "__main__":
    main()
