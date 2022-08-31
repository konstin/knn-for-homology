"""
Major part of this script is in dealing with the fact that there are both domains and proteins involved

After this script seqvec needs to be run manually and then extract_domain_represenation.py once for test and once for
train

External commands:

```bash
scp data/pfam-20-dist/pfam-subset-embed.fasta oculus:pfam-subset-embed.fasta
ssh oculus "/home/kschuetze/.local/bin/seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --layer LSTM1 --id 0"
# seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --layer LSTM1 --id 0
scp oculus:pfam-subset-embed.npz data/pfam-20-dist/pfam-subset-embed.npz
python seqvec_search/extract_domain_representations.py --embeddings data/pfam-20-dist/pfam-subset-embed.npz --domains data/pfam-20-dist/extract_test.json --out test-data/pfam-20-dist/test.npy
python seqvec_search/extract_domain_representations.py --embeddings data/pfam-20-dist/pfam-subset-embed.npz --domains data/pfam-20-dist/extract_train.json --out test-data/pfam-20-dist/train.npy
```
"""

import gzip
import json
import random
import shlex
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from random import Random
from typing import Dict, List

import pandas
from tqdm import tqdm

from seqvec_search.utils import read_fasta, write_fasta

families_to_select = 20
entries_per_family_min = 7
entries_per_family_max = 13


def download_and_extract(url: str, filename: Path):
    with urllib.request.urlopen(url) as fp, filename.open("wb") as target:
        unzipped = gzip.open(fp)
        shutil.copyfileobj(unzipped, target)


def main():
    git_to_root = shlex.split("git rev-parse --show-toplevel")
    project_root = Path(subprocess.check_output(git_to_root, text=True).strip())
    # Reusable files such as pfam
    data = project_root.joinpath("data")
    data.mkdir(exist_ok=True)
    # Specific files such as the full sequence and subset files
    pfam_20_dist = data.joinpath("pfam-20-dist")
    pfam_20_dist.mkdir(exist_ok=True)
    # Determism
    rng = Random(42)

    # Download Pfam A
    if not data.joinpath("Pfam-A.fasta").is_file():
        download_and_extract(
            "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.fasta.gz",
            data.joinpath("Pfam-A.fasta"),
        )

    # Count the family memberships, so we can pick families of an appropriate size
    if not data.joinpath("pfam-sizes.json").is_file():
        with data.joinpath("Pfam-A.fasta").open() as pfam_a_fasta:
            family_sizes = defaultdict(int)
            for line in tqdm(pfam_a_fasta):
                if line.startswith(">"):
                    # >A0A3S4RAV7_9ACAR/192-225 A0A3S4RAV7.1 PF10417.10;1-cysPrx_C;
                    family_sizes[line.split(" ")[-1].split(".")[0]] += 1
        with data.joinpath("pfam-sizes.json").open("w") as family_size_file:
            json.dump(family_sizes, family_size_file)
    else:
        family_sizes = json.loads(data.joinpath("pfam-sizes.json").read_text())

    # Pick random families
    well_sized = list(
        filter(
            lambda item: item[1] > entries_per_family_min + entries_per_family_max,
            family_sizes.items(),
        )
    )
    well_sized = [name for name, size in well_sized]
    families = rng.sample(well_sized, families_to_select)
    print(f"Selected families: {' '.join(families)}")

    # Extract the picked domains
    if not pfam_20_dist.joinpath("pfam-subset.fasta").is_file():
        with data.joinpath(
            "Pfam-A.fasta"
        ).open() as pfam_a_fasta, pfam_20_dist.joinpath("pfam-subset.fasta").open(
            "w"
        ) as subset:
            write = False
            for line in tqdm(pfam_a_fasta):
                if line.startswith(">"):
                    write = line.split(" ")[-1].split(".")[0] in families
                if write:
                    subset.write(line)

    # Download the full sequences for picked families
    for family in tqdm(families):
        url = f"https://pfam.xfam.org/family/{family}/alignment/long/gzipped"
        target = pfam_20_dist.joinpath(f"{family}-full-sequences.fasta")
        if not target.is_file():
            download_and_extract(url, target)

    # Index full sequences
    protein_sequences = dict()
    for family in families:
        fasta_path = pfam_20_dist.joinpath(f"{family}-full-sequences.fasta")
        protein_sequences.update(
            read_fasta(fasta_path, lambda name: name.split(" ")[0])
        )

    # There's a mismatch between the full sequences and Pfam A
    domain_subset = read_fasta(pfam_20_dist.joinpath("pfam-subset.fasta"))
    missing = 0
    total = 0
    for i in list(domain_subset.keys()):
        total += 1
        if i.split("/")[0] not in protein_sequences:
            del domain_subset[i]
            missing += 1
    print(f"Deleted {missing} missing entries of {total} total")

    # Build index over domains
    domains_df = []
    for key, sequence in domain_subset.items():
        [protein, domain_range] = key.split(" ")[0].split("/")
        family = key.split(" ")[-1].split(".")[0]
        domain_range = domain_range.replace("-", ":")
        domains_df.append((family, protein, domain_range, sequence))

    # Split each family
    domain_subset_embed = dict()
    domain_id_to_family: Dict[str, str] = dict()
    domain_subset_test = []
    domain_subset_train = []
    split_sizes = []
    for _, domain_data in groupby(domains_df, lambda x: x[0]):
        domain_data = list(domain_data)
        subset = rng.sample(
            list(domain_data), entries_per_family_min + entries_per_family_max
        )
        for family, protein, domain_range, sequence in subset:
            domain_subset_embed[protein] = protein_sequences[protein]
            domain_id_to_family[f"{protein}/{domain_range}"] = family
        # Pick from a uniform distribution between entries_per_family_min and entries_per_family_max
        split_size = rng.randint(entries_per_family_min, entries_per_family_max)
        domain_subset_test += subset[:split_size]
        domain_subset_train += subset[split_size:]
        split_sizes.append(split_size)
    print(sum(split_sizes), len(split_sizes), " ".join(str(i) for i in split_sizes))

    # Write labels to file
    with pfam_20_dist.joinpath("ids_to_family.json").open("w") as fp:
        json.dump(domain_id_to_family, fp)

    # Write domains for test and train
    with pfam_20_dist.joinpath("test.fasta").open("w") as fp:
        for _, protein, domain_range, sequence in domain_subset_test:
            fp.write(f">{protein}/{domain_range}\n{sequence}\n")
    with pfam_20_dist.joinpath("train.fasta").open("w") as fp:
        for _, protein, domain_range, sequence in domain_subset_train:
            fp.write(f">{protein}/{domain_range}\n{sequence}\n")

    # Write full sequences, to be embeddded and then extracted by extract_domain_representations.py
    write_fasta(pfam_20_dist.joinpath("pfam-subset-embed.fasta"), domain_subset_embed)

    # Write annotations for full sequences
    domain_extract_test: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    domain_extract_train: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for _, protein, domain_range, _ in domain_subset_test:
        domain_extract_test[protein][f"{protein}/{domain_range}"] = [domain_range]
    for _, protein, domain_range, _ in domain_subset_train:
        domain_extract_train[protein][f"{protein}/{domain_range}"] = [domain_range]
    with pfam_20_dist.joinpath("extract_test.json").open("w") as fp:
        json.dump(domain_extract_test, fp)
    with pfam_20_dist.joinpath("extract_train.json").open("w") as fp:
        json.dump(domain_extract_train, fp)

    # Move known good into the tree
    for filename in ["test.fasta", "train.fasta", "ids_to_family.json"]:
        shutil.copy(
            pfam_20_dist.joinpath(filename),
            project_root.joinpath("test-data/pfam-20-dist").joinpath(filename),
        )


if __name__ == "__main__":
    main()
