# %%
import subprocess
import sys
from random import Random

from pfam.proteins_shared import full_sequences_fasta
from pfam.reverse_shared import forward, reverse, shuffle

# %%

random = Random(42)

lines = full_sequences_fasta.read_text().splitlines()
original_full_sequences_ids = [line.split(" ")[1] for line in lines[::2]]
chosen_sample = set(random.sample(original_full_sequences_ids, 10000))

# %%

with forward.open("w") as forward_fp, reverse.open("w") as reverse_fp, shuffle.open(
    "w"
) as shuffle_fp:
    for seq_id, sequence in zip(lines[::2], lines[1::2]):
        if seq_id.split(" ")[1] in chosen_sample:
            forward_fp.write(seq_id + "\n")
            forward_fp.write(sequence + "\n")
            reverse_fp.write(seq_id + "\n")
            reverse_fp.write(sequence[::-1] + "\n")
            shuffle_fp.write(seq_id + "\n")
            shuffle_fp.write("".join(random.sample(sequence, len(sequence))) + "\n")

# %%

for file in [forward, reverse, shuffle]:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pfam.embed_t5_fp16",
            str(file),
            str(file.with_suffix(".npy")),
            "--batch-size",
            str(7000),
        ]
    )
