# Nearest neighbor search on embeddings rapidly identifies distant protein relations

Konstantin Schütze, Michael Heinzinger, Martin Steinegger, Burkhard Rost

Now published in [Frontiers in Bioinformatics](https://www.frontiersin.org/articles/10.3389/fbinf.2022.1033775/full)!

This repository contains the code that generates the datasets, runs the benchmarks and plots all figures. It also contains the each figure's and table's raw data (in `more_sensitive/*/*.h5`).

## Abstract

Since 1992, all state-of-the-art (SOTA) methods for fast and sensitive identification of evolutionary, structural, and functional relations between proteins (also referred to as “homology detection”) use sequences and sequence-profiles (PSSMs). Protein Language Models (pLMs) generalize sequences, possibly capturing the same constraints as PSSMs, e.g., through embeddings. Here, we explored how to use such embeddings for nearest neighbor searches to identify relations between protein pairs with diverged sequences (remote homology detection for levels of <20% pairwise sequence identity, PIDE). While this approach excelled for proteins with single domains, we demonstrated the current challenges applying this to multi-domain proteins and presented some ideas how to overcome existing limitations, in principle. We observed that sufficiently challenging data set separations were crucial to provide deeply relevant insights into the behavior of nearest neighbor search when applied to the protein embedding space, and made all our methods readily available for others

## Setup

Prerequisites: Install python 3.8, poetry 1.1, MMseqs2 13, openblas and openmp on a linux machine (Ubuntu 20.04 is tested). Make sure poetry and mmseqs are in `PATH`

```shell
pip install -U git+https://github.com/konstin/protein-knn
poetry install
curl -L https://github.com/soedinglab/MMseqs2/releases/download/13-45111/mmseqs-linux-avx2.tar.gz | tar xz
```

# Reproducing

## CATH20

```shell
python -m cath.embed-all # Needs GPU
python -m cath.search # Needs CPU
python -m cath.cath # Writes the figures and tables
```

## Pfam20

```shell
python -m pfam.prepare_subset10_full_sequences
python -m pfam.embed_t5_fp16 pfam/subset10/full_sequences.fasta pfam/subset10_full_sequences/full_sequences.npy --batch-size 7000 # Need GPU
python -m pfam.proteins_search hnsw # Needs CPU
python -m pfam.proteins_search flat # Needs CPU
python -m pfam.proteins # Writes the figures and tables
```
