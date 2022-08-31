External commands:
```
seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --layer LSTM1 --id 0
python extract_domain_representations.py --embeddings pfam-subset-embed.npz --domains extract_test.json --out test.npy
python extract_domain_representations.py --embeddings pfam-subset-embed.npz --domains extract_train.json --out train.npy
```

Automation notes

```shell
cat << EOF | ssh oculus "cd bachelorarbeit/storage; ../.venv/bin/python"
from pathlib import Path
families = ["PF07423", "PF09163", "PF16983", "PF17279", "PF05545", "PF16870", "PF08565", "PF01391", "PF18262", "PF06136", "PF12866", "PF00211", "PF13280", "PF06455", "PF03380", "PF09186", "PF18434", "PF14261", "PF04168", "PF15200"]

with Path("Pfam-A.fasta").open() as pfam_a_fasta, Path("pfam-subset.fasta").open("w") as subset:
    write = False
    for line in pfam_a_fasta:
        if line.startswith(">"):
            if line.split(" ")[-1].split(".")[0] in families:
                write = True
            else:
                write = False
        if write:
            subset.write(line)
EOF
scp oculus:bachelorarbeit/storage/pfam-subset.fasta data/pfam-20-10/pfam-subset.fasta

cp data/pfam-20-10/ids_to_family.json test-data/pfam-20-10/ids_to_family.json
scp data/pfam-20-10/pfam-subset-embed.fasta oculus:masterpraktikum/data/pfam-subset-embed.fasta
scp test-data/pfam-20-10/pfam-subset-embed.fasta oculus:masterpraktikum/data/extract_domain_representations.py
scp data/pfam-20-10/extract_test.json oculus:masterpraktikum/data/extract_test.json
scp data/pfam-20-10/extract_train.json oculus:masterpraktikum/data/extract_train.json
ssh oculus "cd masterpraktikum/data; ../.venv/bin/seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --layer LSTM1 --id 0"
ssh oculus "cd masterpraktikum/data; ../.venv/bin/python extract_domain_representations.py --embeddings pfam-subset-embed.npz --domains extract_test.json --out test.npy"
ssh oculus "cd masterpraktikum/data; ../.venv/bin/python extract_domain_representations.py --embeddings pfam-subset-embed.npz --domains extract_train.json --out train.npy"
scp oculus:masterpraktikum/data/train.npy test-data/pfam-20-10/train.npy
scp oculus:masterpraktikum/data/train.json test-data/pfam-20-10/train.json
scp oculus:masterpraktikum/data/test.npy test-data/pfam-20-10/test.npy
scp oculus:masterpraktikum/data/test.json test-data/pfam-20-10/test.json

# Sum all layers; Should be slightly worse
cp test-data/pfam-20-10/* test-data/pfam-20-10-sum
ssh oculus "cd masterpraktikum/data; ../.venv/bin/seqvec -i pfam-subset-embed.fasta -o pfam-subset-embed.npz --id 0"
ssh oculus "cd masterpraktikum/data; ../.venv/bin/python extract_domain_representations.py --embeddings pfam-subset-embed.npz --domains extract_test.json --out test.npy"
ssh oculus "cd masterpraktikum/data; ../.venv/bin/python extract_domain_representations.py --embeddings pfam-subset-embed.npz --domains extract_train.json --out train.npy"
scp oculus:masterpraktikum/data/train.npy test-data/pfam-20-10-sum/train.npy
scp oculus:masterpraktikum/data/train.json test-data/pfam-20-10-sum/train.json
scp oculus:masterpraktikum/data/test.npy test-data/pfam-20-10-sum/test.npy
scp oculus:masterpraktikum/data/test.json test-data/pfam-20-10-sum/test.json
```