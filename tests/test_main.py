from pathlib import Path

import numpy

from seqvec_search import mmseqs
from seqvec_search.data import LoadedData
from seqvec_search.main import faiss_search, evaluate_faiss, evaluate


def test_search_ann():
    data = LoadedData.from_options(path=Path("test-data/small-random"), hits=5)
    queries = numpy.load(str(data.test))
    results, scores, search_time = faiss_search(
        numpy.load(str(data.train)), queries, data.hits
    )
    auc1s, tps = evaluate_faiss(data, results)
    assert auc1s == [1.0, 1 / 3, 2 / 3, 0.0, 0.0, 1 / 3]
    assert tps == [1.0, 2 / 3, 2 / 3, 1.0, 1.0, 1.0]


def test_ann_alignment():
    data = LoadedData.from_options(path=Path("test-data/pfam-20-10"), hits=10)
    queries = numpy.load(str(data.test))
    results, scores, _ = faiss_search(numpy.load(str(data.train)), queries, data.hits)
    auc1s_ann, tps_ann = evaluate_faiss(data, results)
    assert numpy.mean(auc1s_ann) == 0.871
    assert numpy.mean(tps_ann) == 0.91

    mmseqs.write_prefilter_db_data(
        data, numpy.arange(queries.shape[0]), results, scores
    )
    mmseqs.align(data)
    hits = mmseqs.read_result_db(data, data.mmseqs_dir.joinpath("result_combined"))

    # noinspection PyTypeChecker
    auc1s_ann_alignment, tps_ann_alignment = evaluate(data, hits.items())
    assert numpy.mean(auc1s_ann_alignment) == 0.8925
    assert numpy.mean(tps_ann_alignment) == 0.91
