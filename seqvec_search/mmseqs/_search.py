import logging
import time
from subprocess import check_call
from tempfile import TemporaryDirectory

from seqvec_search import mmseqs
from seqvec_search.data import LoadedData
from seqvec_search.utils import E_VALUE_CUTOFF

logger = logging.getLogger(__name__)


def search(
    data: LoadedData, sensitivity: float = 7.5, e_value_cutoff: float = E_VALUE_CUTOFF
) -> float:
    """Calls `mmseqs search`"""
    mmseqs.create_sequence_dbs(data)

    logger.info("Searching with MMseqs2")
    start = time.time()
    # Otherwise MMseqs2 will complain that "result_mmseqs2.dbtype exists already"
    for old_result_file in data.mmseqs_dir.glob("result_mmseqs2*"):
        old_result_file.unlink()
    # usage: mmseqs search <i:queryDB> <i:targetDB> <o:alignmentDB> <tmpDir> [options]
    with TemporaryDirectory() as temp_dir:
        check_call(
            [
                "mmseqs/bin/mmseqs",
                "search",
                "-e",
                str(e_value_cutoff),
                "-s",
                str(sensitivity),
                data.mmseqs_test,
                data.mmseqs_train,
                data.mmseqs_dir.joinpath("result_mmseqs2"),
                temp_dir,
            ]
        )
    total = time.time() - start
    logger.info(f"`mmseqs search` took {total :f}s")
    return total
