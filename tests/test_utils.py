from pathlib import Path

from seqvec_search import create_index
from seqvec_search.utils import read_fasta


def test_read_fasta():
    sequences = read_fasta(Path("test-data/sequences.fasta"), lambda x: x.split(" ")[0])
    assert list(sequences.keys()) == [
        "sp|P00864|CAPP_ECOLI",
        "6U7P:A|PDBID|CHAIN|SEQUENCE",
    ]
    assert len(sequences["sp|P00864|CAPP_ECOLI"]) == 883
    assert len(sequences["6U7P:A|PDBID|CHAIN|SEQUENCE"]) == 99


def test_create_index():
    create_index.main(
        ["--dir", "test-data/pfam-20-10", "--index", "test-data/pfam-20-10/index.bin"]
    )
    assert Path("test-data/pfam-20-10/index.bin").exists()
    # TODO: Search and check AUC1
