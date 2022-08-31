from pfam.pfam_shared import pfam

reverse_data = pfam.joinpath("reverse_data")
reverse_data.mkdir(exist_ok=True)

forward = reverse_data.joinpath("forward.fasta")
reverse = reverse_data.joinpath("reverse.fasta")
shuffle = reverse_data.joinpath("shuffle.fasta")
