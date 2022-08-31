#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy

from seqvec_search.utils import configure_logging

logger = logging.getLogger(__name__)


def main(args: Optional[Sequence[str]] = None):
    configure_logging()
    parser = argparse.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(),
        help="The name of the directory containing the database",
    )
    # noinspection PyTypeChecker
    parser.add_argument(
        "--index", type=Path, required=True, help="The location to write the index to"
    )
    parser.add_argument(
        "--param",
        type=int,
        default=1024,
        help="The tuning parameter of the index. Higher means higher precision",
    )
    args = parser.parse_args(args)

    logger.info(f"Loading database from {args.dir.joinpath('train.npy')}")
    embeddings = numpy.load(str(args.dir.joinpath("train.npy")))
    logger.info(f"Training LSH index with {args.param} bits on {embeddings.shape}")
    lsh_index = faiss.IndexLSH(embeddings.shape[1], args.param)
    # noinspection PyArgumentList
    lsh_index.train(embeddings)
    # noinspection PyArgumentList
    lsh_index.add(embeddings)
    logger.info(f"Writing out the LSH index")
    faiss.write_index(lsh_index, str(args.index))


if __name__ == "__main__":
    main()
