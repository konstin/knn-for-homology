from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import pandas


def main():
    sizes = dict()
    for mmseqs_db in sorted(Path("data/mmseqs_results").glob("*.dbtype")):
        sensitivity = mmseqs_db.name.replace(".dbtype", "")[1:]
        size = 0
        for datafile in Path("data/mmseqs_prefilter").glob(f"s{sensitivity}.?"):
            size += sum(1 for line in datafile.open())
        sizes[float(sensitivity)] = size

    mmseqs_data = pandas.read_csv(
        "data/cath-figures/mmseqs_benchmark.csv", index_col="Sensitivity"
    )
    mmseqs_data["prefilter_size"] = pandas.Series(sizes)

    test_size = len(numpy.load("data/pfam-dist/test.npy"))
    novel_data = pandas.read_csv(
        "data/cath-figures/novel_benchmark.csv", index_col="Hits"
    )

    plt.plot(mmseqs_data["prefilter_size"], mmseqs_data["AUC1"], label="MMseqs2")
    plt.plot(novel_data.index * test_size, novel_data["AUC1"], label="LSH")
    plt.grid()
    plt.ylabel("AUC1")
    plt.xlabel("Number of entries in the prefilter database")
    plt.ylim((0, 1))
    plt.legend()
    plt.savefig("data/cath-figures/prefilter_size_vs_auc1.svg")
    plt.savefig("data/cath-figures/prefilter_size_vs_auc1.jpg")
    plt.close()


if __name__ == "__main__":
    main()
