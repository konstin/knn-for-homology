import sys
from time import time

import faiss
import humanize
import numpy

from pfam.proteins_shared import full_sequences_data


def main():
    index_mode = sys.argv[1]
    # if full_sequences_data.joinpath(
    #    f"full_sequences_{index_mode}_scores.npy"
    # ).is_file():
    #    return
    npy = full_sequences_data.joinpath(f"full_sequences.npy")
    embeddings = numpy.load(npy).astype(numpy.float32)
    print("full_sequences", embeddings.shape)

    start = time()
    faiss.normalize_L2(embeddings)
    if index_mode == "flat":
        index = faiss.IndexFlat(embeddings.shape[1], faiss.METRIC_INNER_PRODUCT)
    elif index_mode == "lsh":
        index = faiss.IndexLSH(embeddings.shape[1], 2048)
    elif index_mode == "hnsw":
        # Index creation took 15s
        # Search took 77s
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 42, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 256
    else:
        raise ValueError(index_mode)
    # noinspection PyArgumentList
    index.train(embeddings)
    # noinspection PyArgumentList
    index.add(embeddings)
    print(f"Index creation took {int(time() - start)}s")
    index_file = full_sequences_data.joinpath(f"full_sequences_{index_mode}.index")
    faiss.write_index(index, str(index_file))
    print(
        f"Embeddings: {humanize.naturalsize(npy.stat().st_size)} "
        f"Index: {humanize.naturalsize(index_file.stat().st_size)} "
        f"Difference: {humanize.naturalsize(index_file.stat().st_size - npy.stat().st_size)}"
    )

    start = time()
    # noinspection PyArgumentList
    flat_scores, flat_hits = index.search(embeddings, 1000)
    print(f"Search took {int(time() - start)}s")
    numpy.save(
        full_sequences_data.joinpath(f"full_sequences_{index_mode}_scores.npy"),
        flat_scores,
    )
    numpy.save(
        full_sequences_data.joinpath(f"full_sequences_{index_mode}_hits.npy"), flat_hits
    )


if __name__ == "__main__":
    main()
