

```shell
python -m pfam.make_slices
# 6:33:58
python -m pfam.embed_t5_fp16 pfam/subset10_full_sequences/slices.fasta pfam/subset10_full_sequences/slices.npy
# 455s (index)
# 5458s (no index)
python -m pfam.slices_search
```

```
bsub -m lsf-server-2 -gpu num=1:mode=shared:j_exclusive=yes -J "Measure exact time required to embed Pfam T5v3 fp16" "python -m pfam.embed_t5_fp16 pfam/subset10_full_sequences/full_sequences.fasta pfam/subset10_full_sequences/full_sequences.npy --batch-size 7000"
```