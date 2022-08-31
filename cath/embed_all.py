"""
Generates embeddings with all embedders and writes them to cath/data/{embedder_name}.npy

We call the embedders in a subprocess to work around
https://github.com/sacdallago/bio_embeddings/issues/40

Additional complication: We don't know yet how to set a good batch size,
but if we set it too low it takes very long but if we set it too high,
some of the transformers will cause the MMU to fault and produce only
errors afterwards, even with single sequence processing
"""

import subprocess
import time

import numpy
from tqdm import tqdm

from cath.cath_shared import cath_data, load_files, fasta_file
from cath.embed import embedder_by_name


def aa_composition():
    embedder_name = "AA Composition"
    if cath_data.joinpath(f"{embedder_name}.npy").is_file():
        print(f"{embedder_name} already done, skipping")
        return
    lines = fasta_file.read_text().splitlines()
    sequences = [line[1:] for line in lines[1::2]]

    start = time.time()

    all_aa = sorted(set("".join(sequences)))
    aa_compositions = []
    for sequence in tqdm(sequences):
        one_hot = [numpy.asarray(all_aa) == i for i in sequence]
        aa_compositions.append(numpy.stack(one_hot).astype(numpy.float32).mean(axis=0))
    aa_compositions = numpy.stack(aa_compositions)

    end = time.time()
    print(f"Embedding with {embedder_name} took {end - start}")
    cath_data.joinpath(f"{embedder_name}.time2.txt").write_text(str(end - start))

    numpy.save(cath_data.joinpath(f"{embedder_name}.npy"), aa_compositions)


def main():
    cath_data.mkdir(exist_ok=True)
    load_files()

    aa_composition()

    for embedder_name in embedder_by_name.keys():
        if cath_data.joinpath(f"{embedder_name}.npy").is_file():
            print(f"{embedder_name} already done, skipping")
            continue
        try:
            start = time.time()
            subprocess.check_call(["python", "-m", "cath.embed", embedder_name])
            end = time.time()
        except subprocess.CalledProcessError as err:
            print(f"Failed to embed with {embedder_name}: {err}")
            continue
        print(f"Embedding with {embedder_name} took {end - start}")
        cath_data.joinpath(f"{embedder_name}.time2.txt").write_text(str(end - start))


if __name__ == "__main__":
    main()
