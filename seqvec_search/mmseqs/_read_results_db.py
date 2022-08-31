import logging
import mmap
from pathlib import Path
from typing import Iterable, List, BinaryIO, Dict, Tuple

import numpy
import pandas
from numpy import ndarray
from tqdm import tqdm

from seqvec_search import mmseqs
from seqvec_search.data import LoadedData

logger = logging.getLogger(__name__)


class MultiMMap:
    """Index multiple memory mapped files as if they were a single contiguous buffer. Use as context manager"""

    files: Iterable[Path]
    mmaps: List[mmap.mmap]
    file_handles: List[BinaryIO]
    sizes: List[int]

    def __init__(self, files: Iterable[Path]):
        self.files = files
        self.file_handles = [file.open("rb") for file in self.files]
        assert len(self.file_handles) > 0, "No files given"
        # noinspection PyArgumentList
        self.mmaps = [
            mmap.mmap(fp.fileno(), 0, prot=mmap.PROT_READ) for fp in self.file_handles
        ]
        self.sizes = [mmaped.size() for mmaped in self.mmaps]

    def __enter__(self) -> "MultiMMap":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mmaped in self.mmaps:
            mmaped.close()
        for file_handle in self.file_handles:
            file_handle.close()

    def __getitem__(self, item: slice) -> bytes:
        assert isinstance(item, slice)
        start = item.start
        stop = item.stop
        for pos, size in enumerate(self.sizes):
            # Is this inside the current file or do we need to check the next file?
            if start < size:
                assert stop <= size, (start, stop, size)
                return self.mmaps[pos][start:stop]
            else:
                start -= size
                stop -= size
        raise IndexError(f"{item}, {self.sizes}")


def read_result_db(data: LoadedData, result_db: Path) -> Dict[str, List[str]]:
    return read_result_db_impl(
        data.train_ids, data.mmseqs_train, data.test_ids, data.mmseqs_test, result_db
    )


def read_result_db_impl(
    train_ids: List[str],
    mmseqs_train: Path,
    test_ids: List[str],
    mmseqs_test: Path,
    result_db: Path,
) -> Dict[str, List[str]]:
    """Reads an MMseqs2 result database and returns the hits for each query

    https://github.com/soedinglab/MMseqs2/wiki#alignment-format"""
    # TODO: Call make_mmseqs_id_map only once
    logger.info("Making a faiss to mmseqs id maps")
    test_mmseqs_to_faiss = numpy.argsort(mmseqs.make_id_map(test_ids, mmseqs_test))
    train_mmseqs_to_faiss = numpy.argsort(mmseqs.make_id_map(train_ids, mmseqs_train))

    logger.info("Reading results")
    index = pandas.read_csv(
        str(result_db) + ".index", sep="\t", names=["query_id", "offset", "record_size"]
    )

    hits: Dict[str, List[str]] = dict()

    # Assume there are less than 100 files
    data_files = sorted(
        list(result_db.parent.glob(f"{result_db.name}.?"))
        + list(result_db.parent.glob(f"{result_db.name}.??")),
        key=lambda x: int(x.name.split(".")[1]),
    )
    with MultiMMap(data_files) as mmseqs_results:
        for query_id, offset, record_size in tqdm(
            index.itertuples(index=False, name=None), total=len(index)
        ):
            """
            # The minus one removes the null byte
            matches_bytes = BytesIO(mmseqs_results[offset : offset + record_size - 1])
            alignment_db_names = [
                "targetID",
                "alnScore",
                "seqIdentity",
                "eVal",
                "qStart",
                "qEnd",
                "qLen",
                "tStart",
                "tEnd",
                "tLen",
            ]
            matches = pandas.read_csv(
                matches_bytes, sep="\t", names=alignment_db_names, usecols=["targetID"]
            )
            target_ids = train_mmseqs_to_faiss[matches["targetID"]]
            """
            # The minus one removes the null byte
            tsv_slice = mmseqs_results[offset : offset + record_size - 1]
            try:
                # The minus one the final newline and ignores empty hits
                matches = [int(x[: x.find(b"\t")]) for x in tsv_slice.split(b"\n")[:-1]]
            except Exception:
                print(tsv_slice)
                raise
            query_id = test_ids[test_mmseqs_to_faiss[query_id]]
            target_ids = train_mmseqs_to_faiss[matches]
            hit_ids = [train_ids[i] for i in target_ids]
            hits[query_id] = hit_ids
    return hits


def read_result_db_with_e_value(
    train_ids: List[str],
    mmseqs_train: Path,
    test_ids: List[str],
    mmseqs_test: Path,
    result_db: Path,
) -> Tuple[Dict[int, ndarray], Dict[int, ndarray]]:
    """Returns int ids instead of string ids"""
    logger.info("Making a faiss to mmseqs id maps")
    test_mmseqs_to_faiss = numpy.argsort(mmseqs.make_id_map(test_ids, mmseqs_test))
    train_mmseqs_to_faiss = numpy.argsort(mmseqs.make_id_map(train_ids, mmseqs_train))

    index = pandas.read_csv(
        str(result_db) + ".index", sep="\t", names=["query_id", "offset", "record_size"]
    )

    hits: Dict[int, ndarray] = dict()
    e_values: Dict[int, ndarray] = dict()

    # There are two cases: For normal search there are bunch of files numbered, for iterated search they get
    # merged into one single files
    if result_db.is_file():
        data_files = [result_db]
    else:
        # Assume there are less than 100 files
        data_files = sorted(
            list(result_db.parent.glob(f"{result_db.name}.?"))
            + list(result_db.parent.glob(f"{result_db.name}.??")),
            key=lambda x: int(x.name.split(".")[1]),
        )
    with MultiMMap(data_files) as mmseqs_results:
        for query_id, offset, record_size in tqdm(
            index.itertuples(index=False, name=None), total=len(index)
        ):
            # The minus one removes the null byte
            tsv_slice = mmseqs_results[offset : offset + record_size - 1]
            # The minus one the final newline and ignores empty hits
            matches = [int(x[: x.find(b"\t")]) for x in tsv_slice.split(b"\n")[:-1]]
            query_id = test_mmseqs_to_faiss[query_id]
            hits[query_id] = train_mmseqs_to_faiss[matches]
            e_values[query_id] = numpy.asarray(
                [float(x.split(b"\t")[3]) for x in tsv_slice.split(b"\n")[:-1]]
            )
    return hits, e_values


def results_to_array(
    hits: Dict[int, ndarray],
    e_values: Dict[int, ndarray],
    sentinel_e_value: float = 100000,
) -> Tuple[ndarray, ndarray]:
    max_hits = max(len(hits) for hits in hits.values())
    mmseqs_hits_array = []
    mmseqs_e_value_array = []
    for i in tqdm(range(len(hits)), total=len(hits)):
        mmseqs_hits_array.append(numpy.pad(hits[i], (0, max_hits - len(hits[i]))))
        # With E<10000, sentinel_e_value=100000 is bigger than all others
        mmseqs_e_value_array.append(
            numpy.pad(
                e_values[i],
                (0, max_hits - len(e_values[i])),
                constant_values=sentinel_e_value,
            )
        )
    return numpy.asarray(mmseqs_hits_array), numpy.asarray(mmseqs_e_value_array)
