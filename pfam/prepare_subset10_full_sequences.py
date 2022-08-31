# %%
from tqdm import tqdm

from pfam.pfam_shared import subset10, pfamseq, pfam, build_domain_ranges

# %%

test = build_domain_ranges(subset10.joinpath("test.fasta"))
train = build_domain_ranges(subset10.joinpath("train.fasta"))
proteins = set(test) | set(train)

# %%

subset10_full_sequences = pfam.joinpath("subset10_full_sequences")
subset10_full_sequences.mkdir(exist_ok=True)
full_sequences_fasta = subset10_full_sequences.joinpath("full_sequences.fasta")

with pfamseq.open() as fp, full_sequences_fasta.open("w") as out:
    also_next = False
    for line in tqdm(fp):
        if also_next:
            out.write(line)
            also_next = False
        if line[0] == ">" and line.split(" ")[1] in proteins:
            out.write(line)
            also_next = True

# %%
