import time
from pathlib import Path
from typing import Tuple

import faiss
import numpy
from numpy import ndarray
from tqdm import tqdm

from cath.cath_shared import cath_data


def search(
    embeddings: ndarray, hits: int = 10, metric=faiss.METRIC_INNER_PRODUCT
) -> Tuple[ndarray, ndarray]:
    """Internally, we search for one more hit because the first hit will be the self hit."""
    if metric == faiss.METRIC_INNER_PRODUCT:
        embeddings = embeddings.copy()
        faiss.normalize_L2(embeddings)
    index = faiss.IndexFlat(embeddings.shape[1], metric)
    # noinspection PyArgumentList
    index.add(embeddings)
    # noinspection PyArgumentList
    scores, results = index.search(embeddings, hits + 1)
    # Remove the self hit
    return results[:, 1:], scores[:, 1:]


def search_and_save(cath_data: Path = cath_data):
    for name, metric in [
        ("Cosine", faiss.METRIC_INNER_PRODUCT),
        ("Euclidean", faiss.METRIC_L2),
    ]:
        print(f"Searching with {name}")
        hits = {}
        scores = {}
        for file_path in tqdm(sorted(cath_data.glob("*.npy"))):
            embeddings = numpy.load(file_path)
            # Cast the fp16 embeddings of half precision model to the fp32 format faiss wants
            embeddings = embeddings.astype(numpy.float32)
            print(file_path.stem, embeddings.shape)
            start = time.time()
            hits[file_path.stem], scores[file_path.stem] = search(
                embeddings, metric=metric
            )
            end = time.time()
            print(end - start)
            cath_data.joinpath(
                file_path.with_suffix(f".{name.lower()}-search-time.txt")
            ).write_text(str(end - start))

        numpy.savez(cath_data.joinpath(f"hits_{name.lower()}.npz"), **hits)
        numpy.savez(cath_data.joinpath(f"scores_{name.lower()}.npz"), **scores)


if __name__ == "__main__":
    search_and_save()
