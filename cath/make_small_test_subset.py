import random

import pandas

from cath.cath_shared import fasta_file, load_files, cath_data

seed = 3


def main():
    load_files()

    lines = fasta_file.read_text().splitlines()
    ids = [line[1:] for line in lines[::2]]
    sequences = [line[1:] for line in lines[1::2]]

    with pandas.HDFStore(str(cath_data.joinpath("cath-mapping.h5"))) as hdf5:
        mapping_df = hdf5["cath-mapping"]

    id_to_seq = dict(zip([i.split("|")[2].split("/")[0] for i in ids], sequences))

    col = []
    for d in mapping_df["domain"].values:
        col.append(id_to_seq.get(d))
    mapping_df["seq"] = col

    q = mapping_df[(mapping_df["seq"].notnull())].groupby(["CATH"]).agg(["count"])
    fam = random.Random(seed).choices(list(q[q[("domain", "count")] >= 10].index), k=10)
    print(fam)

    print(
        mapping_df[(mapping_df["seq"].notnull()) & (mapping_df["CATH"].isin(fam))]
        .groupby(["CATH"])
        .agg(["count"])
    )

    chosen = []
    for i in mapping_df[
        (mapping_df["seq"].notnull()) & (mapping_df["CATH"].isin(fam))
    ].groupby(["CATH"]):
        chosen.append(i[1].sample(10))

    with cath_data.joinpath("small-cath-query.fasta").open("w") as fp:
        for i in chosen:
            for domain, CATH, seq in list(
                i[:5][["domain", "CATH", "seq"]].itertuples(index=False)
            ):
                fp.write(f">{domain}|{CATH}|{len(seq)}\n{seq}\n")

    with cath_data.joinpath("small-cath-db.fasta").open("w") as fp:
        for i in chosen:
            for domain, CATH, seq in list(
                i[5:][["domain", "CATH", "seq"]].itertuples(index=False)
            ):
                fp.write(f">{domain}|{CATH}|{len(seq)}\n{seq}\n")


if __name__ == "__main__":
    main()
