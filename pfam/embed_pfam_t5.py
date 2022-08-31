"""
For T5, we (for now) only try the domains themselves because titin is too big for the GPU and for the RAM
(with very inefficient CPU computation)
"""

import json

import numpy

# noinspection PyUnresolvedReferences
from bio_embeddings.embed import ProtTransT5XLU50 as ProtTransT5XLU50Embedder

# noinspection PyUnresolvedReferences
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5Embedder
from tqdm import tqdm

from pfam.pfam_shared import subset10, subset10_t5


def main():
    embedder = ProtTransT5XLU50Embedder(half_precision_model=True)

    for split_part in ["train", "test"]:
        npy_file = subset10_t5.joinpath(f"{split_part}.npy")
        if npy_file.is_file():
            print(f"{npy_file} exists, skipping")
            continue
        lines = subset10.joinpath(f"{split_part}.fasta").read_text().splitlines()
        ids = [line[1:] for line in lines[::2]]
        sequences = lines[1::2]
        original_order = sequences
        sorting = numpy.argsort([-len(i) for i in sequences])
        sequences = list(numpy.asarray(sequences)[sorting])
        assert original_order == list(numpy.asarray(sequences)[numpy.argsort(sorting)])
        subset10_t5.joinpath(f"{split_part}.json").write_text(json.dumps(ids))

        embeddings = []
        # RTX 8000: 7000: 10GB, 8000: 16.5GB, 9000: also passes < 30GB
        for embedding in tqdm(
            super(ProtTransT5Embedder, embedder).embed_many(sequences, 9000),
            total=len(sequences),
        ):
            embeddings.append(embedder.reduce_per_protein(embedding))
        embeddings = numpy.asarray(embeddings)[numpy.argsort(sorting)]  # Undo sorting
        numpy.save(npy_file, embeddings)


if __name__ == "__main__":
    main()
