import logging
from collections import Counter
from typing import Iterable

import numpy
from matplotlib import pyplot as plt
from numpy import ndarray

from seqvec_search.constants import figure_dir
from seqvec_search.data import LoadedData

logger = logging.getLogger(__name__)


def compute_tps_comulative(data: LoadedData, results: ndarray) -> ndarray:
    # TP cumulative
    logger.info("TP cumulative")
    family_sizes = dict(Counter(data.ids_to_family[i] for i in data.train_ids))
    # Make a TP matrix
    is_correct = []
    tp_counts = []
    for key, row in enumerate(results):
        is_tp_list = (
            numpy.asarray([data.ids_to_family[data.train_ids[hit]] for hit in row])
            == data.ids_to_family[data.test_ids[key]]
        )
        is_correct.append(is_tp_list)
        tp_count = family_sizes[data.ids_to_family[data.test_ids[key]]]
        tp_counts.append(tp_count)
    is_correct = numpy.asarray(is_correct)
    tp_counts = numpy.asarray(tp_counts)
    # Help numpy do the correct comparison
    max_tp_expanded = tp_counts.repeat(is_correct.shape[1]).reshape(is_correct.shape)
    return (is_correct.cumsum(axis=1) / max_tp_expanded).mean(axis=0)


def figure_tp_cumulative(
    names: Iterable[str], tp_cumulatives: Iterable[ndarray], filename: str
):
    numpy.savez(
        str(figure_dir.joinpath(f"{filename}.npz")), **dict(zip(names, tp_cumulatives))
    )
    names_and_tp_cumulative = list(zip(names, tp_cumulatives))
    names_and_tp_cumulative.sort(key=lambda x: -x[1].sum())
    for name, tp_cumulative in names_and_tp_cumulative:
        plt.plot(tp_cumulative, label=name)
    plt.xlabel("Number of hits")
    plt.ylabel("Mean fraction of TP found")
    plt.ylim((0, 1))
    plt.legend()
    plt.grid()
    plt.savefig(str(figure_dir.joinpath(f"{filename}.jpg")))
    plt.savefig(str(figure_dir.joinpath(f"{filename}.svg")))
    plt.show()
