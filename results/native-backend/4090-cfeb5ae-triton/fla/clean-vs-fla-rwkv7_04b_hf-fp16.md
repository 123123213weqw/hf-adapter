# clean-vs-fla-rwkv7_04b_hf-fp16

- status: **outside_thresholds**
- model: /home/ubuntu/codex-run/models-triton-cfeb5ae/rwkv7_04b_hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4090
- code: cfeb5aeca860ce444ebb3515a20cc22f7e2b090b
- FLA: 80e494f6c588e091fc8316b612870df29375c5b8

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999924 | 0.10937500 | 0.02766906 | True |
| b1_t17 | 0.99999836 | 0.15625000 | 0.00994429 | True |
| b1_t128 | 0.99999883 | 0.12500000 | 0.00661671 | False |
| b4_t1 | 0.99999771 | 0.10156250 | 0.02006057 | True |
| b4_t17 | 0.99999794 | 0.21875000 | 0.00975381 | True |
| b4_t128 | 0.99999898 | 0.28125000 | 0.00673588 | False |
| cached_teacher_clean_vs_fla | 0.99999904 | 0.13281250 | 0.00744572 | True |
| clean_cached_vs_clean_full | 1.00000000 | 0.00000000 | 0.00000000 | True |
| fla_cached_vs_fla_full | 0.99999914 | 0.09375000 | 0.00711472 | True |
