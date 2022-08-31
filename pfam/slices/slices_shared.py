from pfam.pfam_shared import pfam

slices_data = pfam.joinpath("slices_data")
slices_fasta = slices_data.joinpath("slices.fasta")
slices_npy = slices_data.joinpath("slices_npy")
slices_db_dir = slices_data.joinpath("dbs")

slice_len = 600
overlap = 200
