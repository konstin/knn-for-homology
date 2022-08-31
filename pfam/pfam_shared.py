import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from seqvec_search.make_pfam_subset import make_pfam_subset
from seqvec_search.utils import download_and_extract

project_root = Path(
    subprocess.check_output(
        shlex.split("git rev-parse --show-toplevel"), text=True
    ).strip()
)
pfam = project_root.joinpath("pfam")
subset10 = pfam.joinpath("subset10")
subset10.mkdir(exist_ok=True)
subset10_t5 = pfam.joinpath("subset10_t5")
subset10_t5.mkdir(exist_ok=True)
pfam_a = pfam.joinpath("Pfam-A.fasta")
pfamseq = pfam.joinpath("pfamseq")
mmseqs_bin = project_root.joinpath("mmseqs/bin/mmseqs")


def load_files():
    if not pfam_a.is_file():
        download_and_extract(
            "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam33.1/Pfam-A.fasta.gz",
            pfam_a,
        )
    if not pfamseq.is_file():
        download_and_extract(
            "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam33.1/pfamseq.gz",
            pfamseq,
        )

    if not subset10.joinpath("full_sequences.fasta"):
        make_pfam_subset(subset10, 2020, pfam_a, pfamseq, 10, 10)


def build_domain_ranges(filename: Path) -> Dict[str, List[Tuple[int, int, str]]]:
    # Header format:
    # >K9RCX8_9CYAN/166-202
    substr_dict = defaultdict(list)
    with filename.open() as fp:
        for line in fp:
            if line[0] == ">":
                header = line.strip()[1:]
                [sequence_id, substr] = header.split("/")
                [start, stop] = substr.split("-")
                substr_dict[sequence_id].append((int(start), int(stop), header))

    return substr_dict
