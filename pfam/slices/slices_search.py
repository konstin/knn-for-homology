from time import time

import faiss
import numpy

from pfam.slices.slices_shared import slices_data


def main():
    # full exhaustive single core: 2540s
    for sequence_set in ["slices", "full_sequences"]:
        if slices_data.joinpath(f"{sequence_set}_scores.npy").is_file():
            continue
        slices_npy = slices_data.joinpath(f"{sequence_set}.npy")
        embeddings = numpy.load(slices_npy).astype(numpy.float32)
        print(sequence_set, embeddings.shape)

        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlat(embeddings.shape[1], faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexLSH(embeddings.shape[1], 1024)
        # noinspection PyArgumentList
        index.train(embeddings)
        # noinspection PyArgumentList
        index.add(embeddings)

        start = time()
        # noinspection PyArgumentList
        flat_scores, flat_hits = index.search(embeddings, 1000)
        print(time() - start)
        numpy.save(slices_data.joinpath(f"{sequence_set}_scores.npy"), flat_scores)
        numpy.save(slices_data.joinpath(f"{sequence_set}_hits.npy"), flat_hits)


if __name__ == "__main__":
    main()
