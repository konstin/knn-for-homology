import logging
from pathlib import Path
from subprocess import check_call

from seqvec_search.data import LoadedData

logger = logging.getLogger(__name__)


def create_db(fasta_file: Path, db_name: Path):
    db_name.parent.mkdir(exist_ok=True)
    check_call(["mmseqs/bin/mmseqs", "createdb", str(fasta_file), str(db_name)])


def create_sequence_dbs(data: LoadedData):
    """Converts the fasta files to mmseqs dbs if not already up to date"""
    mmseqs_dir = data.mmseqs_dir
    mmseqs_dir.mkdir(exist_ok=True)

    for db_name, sequences in [
        ("test", data.test_sequences),
        ("train", data.train_sequences),
    ]:
        # Check if non existent or outdated
        if (
            not mmseqs_dir.joinpath(f"{db_name}.dbtype").is_file()
            or mmseqs_dir.joinpath(f"{db_name}.dbtype").stat().st_mtime
            < data.path.joinpath(f"{db_name}.fasta").stat().st_mtime
        ):
            logger.info(f"Creating mmseqs database for {db_name}")
            create_db(sequences, mmseqs_dir.joinpath(db_name))
