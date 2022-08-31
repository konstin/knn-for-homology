import logging
import time
from subprocess import check_call

from seqvec_search.data import LoadedData
from seqvec_search.utils import E_VALUE_CUTOFF

logger = logging.getLogger(__name__)


def align(data: LoadedData, e_value_cutoff: float = E_VALUE_CUTOFF) -> float:
    """Calls `mmseqs align`"""
    logger.info("Aligning with MMseqs2")
    start = time.time()
    # usage: mmseqs align <i:queryDB> <i:targetDB> <i:resultDB> <o:alignmentDB> [options]
    check_call(
        [
            "mmseqs/bin/mmseqs",
            "align",
            "-e",
            str(e_value_cutoff),
            data.mmseqs_test,
            data.mmseqs_train,
            data.mmseqs_dir.joinpath("prefilter"),
            data.mmseqs_dir.joinpath("result_combined"),
        ]
    )
    total = time.time() - start
    logger.info(f"`mmseqs align` took {total :f}s")
    return total
