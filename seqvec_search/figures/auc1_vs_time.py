import matplotlib.pyplot as plt
import pandas


def main():
    mmseqs_data = pandas.read_csv(
        "data/cath-figures/mmseqs_benchmark.csv", index_col="Sensitivity"
    )
    mmseqs_data["Prefiltering"] = [
        int(i.replace("s", "")) for i in mmseqs_data["Prefiltering"]
    ]
    mmseqs_data["Alignment"] = [
        int(i.replace("s", "")) for i in mmseqs_data["Alignment"]
    ]
    novel_data = pandas.read_csv(
        "data/cath-figures/novel_benchmark.csv", index_col="Hits"
    )
    print(novel_data)

    for measurement in ["AUC1", "TP"]:
        plt.plot(
            mmseqs_data["Prefiltering"] + mmseqs_data["Alignment"],
            mmseqs_data[measurement],
            label="MMseqs2",
        )
        plt.plot(
            novel_data["Prefiltering"] + novel_data["Alignment"],
            novel_data[measurement],
            label="LSH",
        )
        plt.xlabel("time (s)")
        plt.ylabel(measurement)
        plt.ylim((0, 1))
        plt.grid()
        plt.legend()
        plt.savefig(f"data/cath-figures/{measurement.lower()}_vs_time.jpg")
        plt.savefig(f"data/cath-figures/{measurement.lower()}_vs_time.svg")
        plt.close()


if __name__ == "__main__":
    main()
