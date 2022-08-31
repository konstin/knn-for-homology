import json
import shlex
import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from typing import Tuple, Union, Iterable, Dict
from urllib.request import urlretrieve

import h5py
import numpy
import pandas
from numpy import ndarray

git_to_root = shlex.split("git rev-parse --show-toplevel")
try:
    project_root = Path(subprocess.check_output(git_to_root, text=True).strip())
except CalledProcessError:
    project_root = Path()
cath = project_root.joinpath("cath")
figures = project_root.joinpath("more_sensitive/cath-figures")
cath_data = cath.joinpath("data")
cath_dbs = cath.joinpath("dbs")
remapped_fasta = cath_data.joinpath("cath-20-remapped.fasta")
fasta_file = cath_data.joinpath("cath-20.fasta")
domain_list = cath_data.joinpath("cath-domain-list.txt")


def load_files():
    cath_data.mkdir(exist_ok=True)
    prefix = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_2_0/"
    url = prefix + "non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_2_0.fa"
    if not fasta_file.is_file():
        print(f"Downloading {url} to {fasta_file}")
        urlretrieve(url, fasta_file)

    url = prefix + "cath-classification-data/cath-domain-list-v4_2_0.txt"
    if not domain_list.is_file():
        print(f"Downloading {url} to {domain_list}")
        urlretrieve(url, domain_list)


def load_mapping(ids) -> Tuple[dict, ndarray]:
    """
    CATH List File (CLF) Format 2.0
    -------------------------------
    This file format has an entry for each structural entry in CATH.

    Column 1:  CATH domain name (seven characters)
    Column 2:  Class number
    Column 3:  Architecture number
    Column 4:  Topology number
    Column 5:  Homologous superfamily number
    Column 6:  S35 sequence cluster number
    Column 7:  S60 sequence cluster number
    Column 8:  S95 sequence cluster number
    Column 9:  S100 sequence cluster number
    Column 10: S100 sequence count number
    Column 11: Domain length
    Column 12: Structure resolution (Angstroms)
               (999.000 for NMR structures and 1000.000 for obsolete PDB entries)
    """
    if cath_data.joinpath("cath-mapping.h5").is_file():
        with pandas.HDFStore(str(cath_data.joinpath("cath-mapping.h5"))) as hdf5:
            mapping_df = hdf5["cath-mapping"]
    else:
        mapping_df = pandas.read_fwf(
            domain_list,
            comment="#",
            colspecs=[(0, 7), (7, 13), (13, 19), (19, 25), (25, 31)],
            usecols=[0, 1, 2, 3, 4],
            names=["domain", "C", "A", "T", "H"],
        )

        cathcode = (
            mapping_df["C"].astype("str")
            + "."
            + mapping_df["A"].astype("str")
            + "."
            + mapping_df["T"].astype("str")
            + "."
            + mapping_df["H"].astype("str")
        )

        mapping_df["CATH"] = cathcode

        with pandas.HDFStore(str(cath_data.joinpath("cath-mapping.h5"))) as hdf_store:
            hdf_store["cath-mapping"] = mapping_df

    # For some reason, using the dataframe directly is really slow, so we use dicts again
    # Contains the identifiers for each of the 4 levels
    mapping_levels = dict()

    set_ids = set(ids)
    for id, cathcode in mapping_df[["domain", "CATH"]].itertuples(index=False):
        if id in set_ids:
            mapping_levels[id] = tuple(cathcode.rsplit(".", i)[0] for i in range(4))

    mapping_array = numpy.asarray([mapping_levels[i] for i in ids])

    return mapping_levels, mapping_array


def read_ids() -> ndarray:
    """Returns the canonical set of ids from ids.json"""
    return numpy.asarray(
        [
            i.split("|")[2].split("/")[0]
            for i in json.loads(cath_data.joinpath("ids.json").read_text())
        ]
    )


def load_h5(filepath: Union[str, Path], ids: Iterable[str]) -> ndarray:
    """Loads an h5 from bio_embeddings into an ndarray ordered like ids"""
    embedding_dict: Dict[str, ndarray] = dict()
    with h5py.File(filepath) as h5:
        for key, value in h5.items():
            cath_id = value.attrs["original_id"].split("|")[2].split("/")
            embedding_dict[cath_id[0]] = value[:]
    return numpy.asarray([embedding_dict[i] for i in ids])


def h5_to_npy(h5: Union[str, Path], ids: Iterable[str]):
    """Copies an h5 from bio_embeddings into an npy file ordered like ids"""
    numpy.save(Path(h5).with_suffix(".npy"), load_h5(h5, ids))
