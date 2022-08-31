"""
Embed with T5v3 fp16 from longest (cut to 2k) to shortest
"""
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy

# noinspection PyUnresolvedReferences
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("fasta")
    parser.add_argument("npy")
    parser.add_argument("--batch-size", type=int, default=7000)
    args = parser.parse_args()
    fasta = args.fasta
    npy = args.npy
    batch_size = args.batch_size

    embedder = ProtTransT5XLU50Embedder(half_precision_model=True)

    lines = Path(fasta).read_text().splitlines()
    sequences = lines[1::2]
    cutoff_len = 3096
    too_long = sum(len(sequence) > cutoff_len for sequence in sequences)
    print(
        f"Cutting {too_long} of {len(sequences)} ({too_long / len(sequences):.1%}) "
        f"proteins longer than {cutoff_len} amino acids"
    )
    sequences = [sequence[:cutoff_len] for sequence in sequences]
    original_order = sequences
    sorting = numpy.argsort([-len(i) for i in sequences])
    sequences = list(numpy.asarray(sequences)[sorting])
    assert original_order == list(numpy.asarray(sequences)[numpy.argsort(sorting)])

    start = time.time()
    embeddings = []
    # RTX 8000: 7000: 10GB, 8000: 16.5GB, 9000: also passes < 30GB
    for embedding in tqdm(
        embedder.embed_many(sequences, batch_size), total=len(sequences)
    ):
        embeddings.append(embedder.reduce_per_protein(embedding))
    stop = time.time()
    embeddings = numpy.asarray(embeddings)[numpy.argsort(sorting)]  # Undo sorting
    numpy.save(npy, embeddings)
    Path(npy).with_suffix(".time.txt").write_text(str(stop - start))


if __name__ == "__main__":
    main()
