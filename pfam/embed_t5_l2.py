import json
import logging
import sys
import time
from argparse import ArgumentParser
from typing import Dict
from typing import Type

import faiss
import numpy

# noinspection PyUnresolvedReferences
from bio_embeddings.embed import ProtTransT5XLU50Embedder, EmbedderInterface

# noinspection PyUnresolvedReferences
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5Embedder
from tqdm import tqdm

from cath.cath_shared import load_files, fasta_file, cath_data

embedder_by_name: Dict[str, Type[EmbedderInterface]] = {
    "ProtT5 XL U50 L2": ProtTransT5XLU50Embedder
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "embedder_name",
        choices=sorted(embedder_by_name.keys()),
        nargs="?",
        default="ProtT5 XL U50 L2",
    )
    args = parser.parse_args([])
    embedder_name = args.embedder_name
    embedder_class = embedder_by_name[embedder_name]

    # Download all required files
    load_files()

    cath_data.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(
        filename=str(cath_data.joinpath(f"{embedder_name}.log"))
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers
    )
    logger = logging.getLogger(__name__)

    lines = fasta_file.read_text().splitlines()
    ids = [line[1:] for line in lines[::2]]
    sequences = lines[1::2]
    cath_data.joinpath("ids.json").write_text(json.dumps(ids))

    logger.info(f"Embedding {embedder_name} with {embedder_class.name}")

    embedder = embedder_class()
    start = time.time()
    embedding_generator = super(ProtTransT5Embedder, embedder).embed_many(
        sequences, None
    )
    embeddings = []
    for i in tqdm(embedding_generator, total=len(sequences)):
        i = i.astype(numpy.float32)
        faiss.normalize_L2(i)
        embeddings.append(embedder.reduce_per_protein(i))
    end = time.time()
    cath_data.joinpath(f"{embedder_name}.time1.txt").write_text(str(end - start))

    embeddings = numpy.asarray(embeddings)
    logger.info(f"Done embedding {embedder_class} {embeddings.shape}")
    numpy.save(cath_data.joinpath(f"{embedder_name}.npy"), embeddings)


if __name__ == "__main__":
    main()
