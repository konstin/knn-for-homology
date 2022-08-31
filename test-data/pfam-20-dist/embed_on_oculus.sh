set -e

cd $(git rev-parse --show-toplevel)

scp data/pfam-20-dist/pfam-subset-embed.fasta oculus:pfam-subset-embed.fasta
ssh oculus "/home/kschuetze/.local/bin/seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --layer LSTM1 --id 0"
# seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --layer LSTM1 --id 0
scp oculus:pfam-subset-embed.npz data/pfam-20-dist/pfam-subset-embed.npz
python seqvec_search/extract_domain_representations.py --embeddings data/pfam-20-dist/pfam-subset-embed.npz --domains data/pfam-20-dist/extract_test.json --out test-data/pfam-20-dist/test.npy
python seqvec_search/extract_domain_representations.py --embeddings data/pfam-20-dist/pfam-subset-embed.npz --domains data/pfam-20-dist/extract_train.json --out test-data/pfam-20-dist/train.npy
