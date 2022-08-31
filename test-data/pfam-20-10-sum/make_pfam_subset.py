# %%
import gzip
import json
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
entries_per_family = 10


def download_and_extract(url: str, filename: Path):
    with urllib.request.urlopen(url) as fp, filename.open("wb") as target:
        unzipped = gzip.open(fp)
        shutil.copyfileobj(unzipped, target)


# %%


def main():
    # %%
    project_root = subprocess.check_output(
        shlex.split("git rev-parse --show-toplevel"), text=True
    ).strip()
    data = Path(project_root).joinpath("data").joinpath("pfam-20-10")
    data.mkdir(exist_ok=True)
    # Pick random families
    tsv_file = data.joinpath("Pfam-A.clans.tsv")
    if not tsv_file.is_file():
        url = (
            "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"
        )
        download_and_extract(url, tsv_file)
    pfam = pandas.read_csv(
        tsv_file,
        delimiter="\t",
        names=["family", "clan", "unnamed1", "unnamed2", "description"],
    )
    families = list(pfam.sample(families_to_select, random_state=24)["family"])
    print(f"Selected families: {' '.join(families)}")

    # %%

    # scp

    # Missing: Download Pfam-A.fasta from ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.fasta.gz

    # Extract the domains
    if not data.joinpath("pfam-subset.fasta").is_file():
        with data.joinpath("Pfam-A.fasta").open() as pfam_a_fasta, data.joinpath(
            "pfam-subset.fasta"
        ).open("w") as subset:
            write = False
            for line in pfam_a_fasta:
                if line.startswith(">"):
                    if line.split(" ")[-1].split(".")[0] in families:
                        write = True
                    else:
                        write = False
                if write:
                    subset.write(line)

    # Download the full sequences
    for family in tqdm(families):
        url = f"https://pfam.xfam.org/family/{family}/alignment/long/gzipped"
        target = data.joinpath(f"{family}-full-sequences.fasta")
        if not target.is_file():
            download_and_extract(url, target)

    family_to_sequences = dict()
    for family in families:
        fasta_path = data.joinpath(f"{family}-full-sequences.fasta")
        family_to_sequences[family] = read_fasta(
            fasta_path, lambda name: name.split(" ")[0]
        )

    domain_subset = read_fasta(data.joinpath("pfam-subset.fasta"))
    protein_sequences = dict()
    for family in families:
        fasta_path = data.joinpath(f"{family}-full-sequences.fasta")
        protein_sequences.update(
            read_fasta(fasta_path, lambda name: name.split(" ")[0])
        )

    missing = 0
    for i in list(domain_subset.keys()):
        if i.split("/")[0] not in protein_sequences:
            del domain_subset[i]
            missing += 1
    print(f"Deleted {missing} missing entries")

    # %%

    df = []
    for key, sequence in domain_subset.items():
        [protein, domain_range] = key.split(" ")[0].split("/")
        family = key.split(" ")[-1].split(".")[0]
        domain_range = domain_range.replace("-", ":")
        df.append((family, protein, domain_range, sequence))

    # %%

    domain_subset_embed = dict()
    domain_id_to_family: Dict[str, str] = dict()
    domain_subset_test = []
    domain_subset_train = []
    for _, domain_data in groupby(df, lambda x: x[0]):
        subset = Random(42).sample(list(domain_data), entries_per_family * 2)
        for family, protein, domain_range, sequence in subset:
            domain_subset_embed[protein] = protein_sequences[protein]
            domain_id_to_family[f"{protein}/{domain_range}"] = family
        domain_subset_test += subset[:entries_per_family]
        domain_subset_train += subset[entries_per_family:]

    write_fasta(data.joinpath("pfam-subset-embed.fasta"), domain_subset_embed)

    with data.joinpath("test.fasta").open("w") as fp:
        for _, protein, domain_range, sequence in domain_subset_test:
            fp.write(f">{protein}/{domain_range}\n{sequence}\n")
    with data.joinpath("train.fasta").open("w") as fp:
        for _, protein, domain_range, sequence in domain_subset_train:
            fp.write(f">{protein}/{domain_range}\n{sequence}\n")

    domain_extract_test: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for _, protein, domain_range, _ in domain_subset_test:
        domain_extract_test[protein][f"{protein}/{domain_range}"] = [domain_range]
    domain_extract_train: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for _, protein, domain_range, _ in domain_subset_train:
        domain_extract_train[protein][f"{protein}/{domain_range}"] = [domain_range]
    with data.joinpath("extract_test.json").open("w") as fp:
        json.dump(domain_extract_test, fp)
    with data.joinpath("extract_train.json").open("w") as fp:
        json.dump(domain_extract_train, fp)
    with data.joinpath("ids_to_family.json").open("w") as fp:
        json.dump(domain_id_to_family, fp)

    # %%


#

if __name__ == "__main__":
    main()
