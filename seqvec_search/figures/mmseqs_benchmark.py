#!/usr/bin/env python3
import logging
from pathlib import Path

import numpy
import pandas

from seqvec_search import mmseqs
from seqvec_search.constants import figure_dir
from seqvec_search.data import LoadedData
from seqvec_search.main import make_figure, evaluate
from seqvec_search.utils import configure_logging

logger = logging.getLogger(__name__)


class MMseqsLoadedData(LoadedData):
    @property
    def mmseqs_test(self) -> Path:
        return Path("data/mmseqs_dbs/test")

    @property
    def mmseqs_train(self) -> Path:
        return Path("data/mmseqs_dbs/train")


def main():
    configure_logging()
    results = []
    for mmseqs_db in sorted(Path("data/mmseqs_results").glob("*.dbtype")):
        sensitivity = mmseqs_db.name.replace(".dbtype", "")[1:]
        logger.info(f"Evaluating MMseqs2 s={sensitivity}")
        data = MMseqsLoadedData.from_options(Path("data/pfam-dist"))
        hits = mmseqs.read_result_db(data, mmseqs_db.with_suffix(""))
        mmseqs_time = int(
            mmseqs_db.parent.joinpath(f"s{sensitivity}-time.txt").read_text().strip()
        )
        # noinspection PyTypeChecker
        auc1s_mmseqs2, tps_mmseqs2 = evaluate(data, hits.items())
        results.append((sensitivity, auc1s_mmseqs2, tps_mmseqs2))
        logger.info(
            f"Mean AUC1 for MMseqs2 s={sensitivity}: {numpy.mean(auc1s_mmseqs2):f}, "
            f"Mean TP: {numpy.mean(tps_mmseqs2):f}, "
            f"Time {mmseqs_time}s"
        )

    results.sort(key=lambda x: -float(x[0]))
    sensitivities, auc1s_mmseqs2s, tps_mmseqs2s = zip(*results)

    make_figure(
        figure_dir,
        auc1s_mmseqs2s,
        sensitivities,
        "AUC1",
        "mmseqs_sensitivities.jpg",
        svg=True,
    )

    rows = []
    for sensitivity, auc1s_mmseqs2, tps_mmseqs2 in results:
        prefiltering = int(
            Path(f"data/mmseqs_prefilter/s{sensitivity}-time.txt").read_text().strip()
        )
        alignment = int(
            Path(f"data/mmseqs_align/s{sensitivity}-time.txt").read_text().strip()
        )
        # noinspection PyTypeChecker
        rows.append(
            (
                round(numpy.mean(auc1s_mmseqs2), 10),
                round(numpy.mean(tps_mmseqs2), 10),
                f"{prefiltering}s",
                f"{alignment}s",
            )
        )

    df = pandas.DataFrame(
        rows, columns=["AUC1", "TP", "Prefiltering", "Alignment"], index=sensitivities
    )
    df.index.name = "Sensitivity"
    Path("data/cath-figures/mmseqs_benchmark.csv").write_text(df.to_csv())
    markdown = df.to_markdown(floatfmt=("g", ".3f", ".3f", "", ""))
    print(markdown)
    Path("data/cath-figures/mmseqs_benchmark.md").write_text(markdown)


if __name__ == "__main__":
    main()
