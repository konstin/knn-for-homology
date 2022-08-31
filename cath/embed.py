import json
import logging
import sys
import time
from argparse import ArgumentParser
from typing import Dict
from typing import Type

import numpy

# noinspection PyUnresolvedReferences
from bio_embeddings.embed import (
    CPCProtEmbedder,
    ESMEmbedder,
    ESM1bEmbedder,
    PLUSRNNEmbedder,
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    ProtTransXLNetUniRef100Embedder,
    ProtTransT5BFDEmbedder,
    ProtTransT5UniRef50Embedder,
    ProtTransT5XLU50Embedder,
    SeqVecEmbedder,
    UniRepEmbedder,
    EmbedderInterface,
)

# noinspection PyUnresolvedReferences
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5Embedder
from tqdm import tqdm

from cath.cath_shared import load_files, fasta_file, cath_data

embedder_by_name: Dict[str, Type[EmbedderInterface]] = {
    "CPCProt": CPCProtEmbedder,
    "ESM": ESMEmbedder,
    "ESM1b": ESM1bEmbedder,
    "PLUS": PLUSRNNEmbedder,
    "ProtAlbert BFD": ProtTransAlbertBFDEmbedder,
    "ProtBert BFD": ProtTransBertBFDEmbedder,
    "ProtXLNet UniRef100": ProtTransXLNetUniRef100Embedder,
    "ProtT5-BFD": ProtTransT5BFDEmbedder,
    "ProtT5 XL U50": ProtTransT5XLU50Embedder,
    "SeqVec": SeqVecEmbedder,
    "UniRep": UniRepEmbedder,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("embedder_name", choices=sorted(embedder_by_name.keys()))
    args = parser.parse_args()
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

    if embedder_class == ESM1bEmbedder:
        # https://github.com/facebookresearch/esm/issues/49
        sequences = [sequence[:1022] for sequence in sequences]

    embedder = embedder_class()
    start = time.time()
    embedding_generator = embedder.embed_many(
        sequences, None
    )  # TODO: Batch size and the cluster
    embeddings = []
    for i in tqdm(embedding_generator, total=len(sequences)):
        if embedder_class == SeqVecEmbedder:
            embeddings.append(i.mean(1))
        else:
            embeddings.append(embedder.reduce_per_protein(i))
    end = time.time()
    cath_data.joinpath(f"{embedder_name}.time1.txt").write_text(str(end - start))

    embeddings = numpy.asarray(embeddings)
    logger.info(f"Done embedding {embedder_class} {embeddings.shape}")
    if embedder_class == SeqVecEmbedder:
        # So SeqVec has three layer, and we want to do different stuff with them
        numpy.save(cath_data.joinpath("SeqVec Sum.npy"), embeddings.sum(1))
        numpy.save(cath_data.joinpath("SeqVec CharCNN.npy"), embeddings[:, 0])
        numpy.save(cath_data.joinpath("SeqVec LSTM1.npy"), embeddings[:, 1])
        numpy.save(cath_data.joinpath("SeqVec LSTM2.npy"), embeddings[:, 2])
    else:
        numpy.save(cath_data.joinpath(f"{embedder_name}.npy"), embeddings)


if __name__ == "__main__":
    main()
