import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from seqvec_search.constants import default_hits


@dataclass
class LoadedData:
    path: Path
    train: Path
    train_ids: List[str]
    knn_index: Optional[Path]
    test: Path
    test_ids: List[str]
    ids_to_family: Dict[str, str]
    train_sequences: Path
    test_sequences: Path
    hits: int = default_hits

    @property
    def mmseqs_dir(self) -> Path:
        return self.path.joinpath("mmseqs_dbs")

    @property
    def mmseqs_test(self) -> Path:
        return self.path.joinpath("mmseqs_dbs").joinpath("test")

    @property
    def mmseqs_train(self) -> Path:
        return self.path.joinpath("mmseqs_dbs").joinpath("train")

    @classmethod
    def from_options(
        cls, path: Path, hits: int = default_hits, knn_index: Optional[Path] = None
    ) -> "LoadedData":
        data = cls(
            path=path,
            train=path.joinpath("train.npy"),
            train_ids=json.loads(path.joinpath("train.json").read_text()),
            knn_index=knn_index,
            test=path.joinpath("test.npy"),
            test_ids=json.loads(path.joinpath("test.json").read_text()),
            ids_to_family=json.loads(path.joinpath("ids_to_family.json").read_text()),
            train_sequences=path.joinpath("train.fasta"),
            test_sequences=path.joinpath("test.fasta"),
            hits=hits,
        )
        return data
