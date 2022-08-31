# %%
from tqdm import tqdm

from pfam.proteins_shared import full_sequences_fasta
from pfam.slices.slices_shared import slices_fasta, overlap, slice_len

# %%

lines = full_sequences_fasta.read_text().splitlines()
ids = lines[::2]
sequences = lines[1::2]


# %%


def main():
    counter = 0
    with slices_fasta.open("w") as fp:
        for sequence_id, sequence in zip(ids, tqdm(sequences)):
            sequence_id = sequence_id.split(" ")[1]
            # max(200) prevents us from skipping <200 proteins
            for start in range(
                0, max(200, len(sequence) - overlap), slice_len - overlap
            ):
                fp.write(f">{sequence_id}-{start}\n")
                fp.write(sequence[start : start + slice_len] + "\n")
                counter += 1
    print(f"Made {counter} slices")


if __name__ == "__main__":
    main()
