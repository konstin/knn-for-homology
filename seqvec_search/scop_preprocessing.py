# %%

# Download the SCOP 2 metadata
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy
import requests
from tqdm import tqdm

scop2path = Path("data/scop-cla-latest.txt")
if not scop2path.is_file():
    response = requests.get("http://scop.mrc-lmb.cam.ac.uk/files/scop-cla-latest.txt")
    response.raise_for_status()
    scop2path.write_text(response.text)


# %%

# Get the uniprot ids


@dataclass
class SCOPCLA:
    protein_type: str
    protein_class: str
    fold: str
    superfamily: str
    family: str

    @staticmethod
    def from_scop_cla(field: str) -> "SCOPCLA":
        """
        From http://scop.mrc-lmb.cam.ac.uk/download:

        SCOPCLA - SCOP domain classification. The abbreviations denote:
        TP=protein type, CL=protein class, CF=fold, SF=superfamily, FA=family
        """
        entries = dict(i.split("=") for i in field.split(","))
        return SCOPCLA(
            protein_type=entries["TP"],
            protein_class=entries["CL"],
            fold=entries["CF"],
            superfamily=entries["SF"],
            family=entries["FA"],
        )


uniprot_ids = defaultdict(dict)
domain_annotations: Dict[str, SCOPCLA] = dict()

for line in scop2path.read_text().splitlines():
    if line.startswith("#"):
        continue
    split = line.split(" ")

    domain_annotations[split[0]] = SCOPCLA.from_scop_cla(split[10])

    # handle non-contigous domains
    ranges = []
    for prot_range in split[4].split(","):
        start, end = prot_range.split("-")
        # We want 0-based indexing
        ranges.append((int(start) - 1, int(end)))
    uniprot_ids[split[3]][split[0]] = ranges

# %%

# Download everything from uniprot

ids = list(uniprot_ids.keys())
response = requests.post(
    "https://www.uniprot.org/uploadlists/",
    files={"file": " ".join(ids)},
    data={"format": "fasta", "from": "ACC+ID", "to": "ACC"},
)
response.raise_for_status()

Path("storage/scop2sequences.fasta").write_text(response.text)

# %%

data: dict = numpy.load("data/scop2sequences.npz")
representations = dict()
for key, value in tqdm(data.items()):
    key = key.split("|")[1]
    for domain_id, domain_range in uniprot_ids[key].items():
        if len(domain_range) == 1:
            domain_embedding = value[
                domain_range[0][0] : domain_range[0][1], 1024:2048
            ].mean(axis=0)
        else:
            domain_parts = []
            for start, end in domain_range:
                domain_parts.append(value[start:end, 1024:2048])
            domain_embedding = numpy.concatenate(domain_parts).mean(axis=0)
        representations[domain_id] = domain_embedding

all_representations = numpy.stack(list(representations.values()))
all_domain_ids = list(representations.keys())
with open("data/scop2embeddings.json", "w") as fp:
    json.dump(all_domain_ids, fp)
numpy.save("data/scop2embeddings.npy", all_representations)

# %%

scop2sequences_file = Path("data/scop2sequences.fasta")
scop2sequences = {}
lines = iter(scop2sequences_file.read_text().splitlines())
current_id = next(lines).split(" ")[0].split("|")[1]
current_seq = ""
for line in lines:
    if line.startswith(">"):
        scop2sequences[current_id] = current_seq
        current_seq = ""
        current_id = line.split(" ")[0].split("|")[1]
    else:
        current_seq += line

found = 0
missing = 0
domains = dict()
for prot_id, ranges in uniprot_ids.items():
    if prot_id not in scop2sequences:
        missing += 1
        continue
    else:
        found += 1
    for domain_id, domain_range in ranges.items():
        domains[domain_id] = "".join(
            scop2sequences[prot_id][start:stop] for start, stop in domain_range
        )
print(f"Missed {missing} of {len(uniprot_ids)} total")

# %%
