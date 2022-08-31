# %%

# UniProt Release 2020_05 UniRef
from pathlib import Path
from typing import Tuple

import faiss
import numpy
from numpy import ndarray

from pfam.pfam_shared import subset10, subset10_t5


def load_embeddings(embedding_set: Path) -> Tuple[ndarray, ndarray]:
    train_embeddings = numpy.load(embedding_set.joinpath("train.npy"))
    test_embeddings = numpy.load(embedding_set.joinpath("test.npy"))
    train_embeddings = train_embeddings.astype(numpy.float32)
    faiss.normalize_L2(train_embeddings)
    test_embeddings = test_embeddings.astype(numpy.float32)
    faiss.normalize_L2(test_embeddings)
    return train_embeddings, test_embeddings


def search_index(embedding_set: Path):
    train_embeddings, test_embeddings = load_embeddings(embedding_set)
    if not embedding_set.joinpath("index_lsh_1024.bin").is_file():
        lsh_index = faiss.IndexLSH(train_embeddings.shape[1], 1024)
        # noinspection PyArgumentList
        lsh_index.train(train_embeddings)
        # noinspection PyArgumentList
        lsh_index.add(train_embeddings)
        faiss.write_index(lsh_index, str(embedding_set.joinpath("index_lsh_1024.bin")))
    else:
        lsh_index = faiss.read_index(str(embedding_set.joinpath("index_lsh_1024.bin")))

    # noinspection PyArgumentList
    index_scores, index_hits = lsh_index.search(test_embeddings, 1000)
    numpy.save(embedding_set.joinpath("index_scores.npy"), index_scores)
    numpy.save(embedding_set.joinpath("index_hits.npy"), index_hits)


def search_flat(embedding_set: Path):
    train_embeddings, test_embeddings = load_embeddings(embedding_set)
    lsh_index = faiss.IndexFlat(train_embeddings.shape[1], faiss.METRIC_INNER_PRODUCT)
    # noinspection PyArgumentList
    lsh_index.train(train_embeddings)
    # noinspection PyArgumentList
    lsh_index.add(train_embeddings)

    # noinspection PyArgumentList
    flat_scores, flat_hits = lsh_index.search(test_embeddings, 1000)
    numpy.save(embedding_set.joinpath("flat_scores.npy"), flat_scores)
    numpy.save(embedding_set.joinpath("flat_hits.npy"), flat_hits)


def main():
    for embedding_set in [subset10_t5, subset10]:
        print(embedding_set, "index")
        search_index(embedding_set)
        print(embedding_set, "flat")
        search_flat(embedding_set)


if __name__ == "__main__":
    main()
