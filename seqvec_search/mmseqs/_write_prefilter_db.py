#!/usr/bin/env python3
"""
Searches with index/metric and then writes out a prefilter db that mmseqs can read
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy
from numpy import ndarray
from tqdm import tqdm

from seqvec_search import mmseqs
from seqvec_search.data import LoadedData

logger = logging.getLogger(__name__)


def make_id_map(ids: List[str], mmseqs_db: Path) -> ndarray:
    """Creates a map from this programs internal (faiss) id to mmseqs' internal id"""
    mmseqs_map: Dict[str, int] = dict()
    with mmseqs_db.with_suffix(".lookup").open() as fp:
        for line in fp:
            seq_mmseqs_id, seq_name, _ = line.split("\t")
            mmseqs_map[seq_name] = int(seq_mmseqs_id)
    faiss_to_mmseqs = numpy.zeros(len(ids), dtype=numpy.int)
    for pos, entry_id in enumerate(ids):
        faiss_to_mmseqs[pos] = mmseqs_map[entry_id]
    return faiss_to_mmseqs


def write_prefilter_db_data(
    data: LoadedData, queries: ndarray, hits: ndarray, scores: ndarray
):
    """Takes the hits and writes a prefilter db for mmseqs to consume.

    Requires the train and test mmseqs db to be written already."""
    mmseqs.create_sequence_dbs(data)

    logger.info("Making a faiss to mmseqs id map")
    test_faiss_to_mmseqs = make_id_map(data.test_ids, data.mmseqs_test)
    train_faiss_to_mmseqs = make_id_map(data.train_ids, data.mmseqs_train)

    prefilter_db = data.mmseqs_dir.joinpath("prefilter")

    write_prefilter_db(
        hits, prefilter_db, queries, scores, test_faiss_to_mmseqs, train_faiss_to_mmseqs
    )


def write_prefilter_db(
    hits: ndarray,
    prefilter_db: Path,
    queries: ndarray,
    scores: ndarray,
    test_faiss_to_mmseqs: ndarray,
    train_faiss_to_mmseqs: ndarray,
    clip: bool = True,
):
    logger.info("Writing prefilter")
    if numpy.sum(hits == -1) > 0:
        logger.warning(f"There are {numpy.sum(hits == -1)} missing hits")

    # Fixed db type for prefilter
    prefilter_db.with_suffix(".dbtype").write_bytes(b"\x07\x00\x00\x00")
    # For simplicity we only create one data file
    data_file = prefilter_db.with_suffix(".0")
    index_file = prefilter_db.with_suffix(".index")
    with data_file.open("wb") as database, index_file.open("wb") as index:
        offset = 0
        # This is much faster for some reason
        # The clip avoids "overflow encountered in multiply" with highly negative values
        # For pfam it takes overly much ram though so I had to deactivate it
        if clip:
            scores_int = numpy.clip(scores, -(10**30), 10**30) * 100
        else:
            scores_int = scores * 100
        for query, hit_entry, score_entry in zip(tqdm(queries), hits, scores_int):
            length = 0
            for hit, score in zip(hit_entry, score_entry):
                if hit == -1:
                    continue
                hit_translated = train_faiss_to_mmseqs[hit]
                # MMseqs doesn't filter on the score if we don't tell it to, so we set
                # an int transformed version of the distance
                # Diagonal is set to 0
                line_bytes = f"{hit_translated}\t{int(score)}\t0\n".encode()
                length += len(line_bytes)
                database.write(line_bytes)
            # Apparently every section is null delimited
            database.write(b"\0")
            length += 1
            # The index includes the length of the entries, so we have to write it after having written the DB
            query_translated = test_faiss_to_mmseqs[query]
            index.write(f"{query_translated}\t{offset}\t{length}\n".encode())
            offset += length
