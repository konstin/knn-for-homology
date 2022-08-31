import gzip
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Callable, TypeVar

import matplotlib
import numpy
import tqdm
from matplotlib import pyplot as plt
from numpy import ndarray

E_VALUE_CUTOFF: float = 10000.0

# https://stackoverflow.com/a/48110626/3549270
# This is run when importing endfig
matplotlib.rcParams["svg.hashsalt"] = 42

tsv_names = [
    "queryID",
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


class TqdmLoggingHandler(logging.Handler):
    """https://stackoverflow.com/a/38739634/3549270"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
            raise


def configure_logging():
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[TqdmLoggingHandler()],
    )


T = TypeVar("T")


def read_fasta(source: Path, rename: Callable[[str], T] = lambda x: x) -> Dict[T, str]:
    sequences: Dict[str, str] = dict()
    sequence_id = None
    acc = None
    with source.open() as fp:
        for line in fp:
            if line[0] == ">":
                if acc is not None:
                    sequences[sequence_id] = acc
                sequence_id = rename(line[1:].strip())
                acc = ""
            else:
                acc += line.strip()
        sequences[sequence_id] = acc
    return sequences


def write_fasta(target: Path, data: Dict[str, str]):
    with target.open("w") as fp:
        for name, sequence in data.items():
            fp.write(f">{name}\n")
            fp.write(sequence + "\n")


def endfig(base_dir: Path, name: str):
    # https://matplotlib.org/2.1.1/users/whats_new.html#reproducible-ps-pdf-and-svg-output
    # https://gitedu.hesge.ch/dimitri.lizzi/bootiful/-/commit/d78d6c9cc94390ad27e9a9c6138430fe52f824d7#73823aec38031f8f9df6a564ab5a9e9aded1c58d
    plt.savefig(str(base_dir.joinpath(name + ".svg")), metadata={"Date": ""})
    plt.savefig(
        str(base_dir.joinpath(name + ".jpg")), dpi=600, pil_kwargs=dict(quality=85)
    )
    plt.savefig(str(base_dir.joinpath(name + ".eps")))
    if matplotlib.get_backend() == "module://backend_interagg":
        plt.show()
    else:
        plt.close()


def rolling_mean(data: ndarray, window_size: int) -> ndarray:
    # https://stackoverflow.com/q/13728392/3549270
    # I had to remove the pandas solution since it rounded low E-values to 0
    # return (
    #     pandas.Series(data)
    #     .rolling(window=window_size)
    #     .mean()
    #     .iloc[window_size - 1 :]
    #     .values
    # )
    return numpy.convolve(data, numpy.ones(window_size) / window_size, mode="valid")


def download_and_extract(url: str, filename: Path):
    with urllib.request.urlopen(url) as fp, filename.open("wb") as target:
        unzipped = gzip.open(fp)
        shutil.copyfileobj(unzipped, target)
