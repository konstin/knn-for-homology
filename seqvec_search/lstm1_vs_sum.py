from pathlib import Path

import numpy

from seqvec_search.data import LoadedData
from seqvec_search.main import faiss_search, make_figure, evaluate_faiss


def main():
    data = LoadedData.from_options(path=Path("test-data/pfam-20-10"))
    results, _, _ = faiss_search(
        numpy.load(str(data.train)), numpy.load(str(data.test)), data.hits
    )
    auc1s1, _ = evaluate_faiss(data, results)

    data = LoadedData.from_options(path=Path("test-data/pfam-20-10-sum"))
    results, _, _ = faiss_search(
        numpy.load(str(data.train)), numpy.load(str(data.test)), data.hits
    )
    auc1s2, _ = evaluate_faiss(data, results)

    make_figure(
        Path("data"),
        [auc1s1, auc1s2],
        ["LSTM1", "SUM"],
        "AUC1",
        "auc1_lstm1_vs_sum.jpg",
    )


if __name__ == "__main__":
    main()
