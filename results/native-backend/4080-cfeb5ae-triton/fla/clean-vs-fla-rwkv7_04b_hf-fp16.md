# clean-vs-fla-rwkv7_04b_hf-fp16

- status: **outside_thresholds**
- model: /home/wzu/codex-run/models/native-v010-triton-cfeb5ae-v2/rwkv7_04b_hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: cfeb5aeca860ce444ebb3515a20cc22f7e2b090b
- FLA: 80e494f6c588e091fc8316b612870df29375c5b8

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999903 | 0.10937500 | 0.02297930 | True |
| b1_t17 | 0.99999884 | 0.12500000 | 0.00916374 | True |
| b1_t128 | 0.99999911 | 0.09375000 | 0.00648224 | False |
| b4_t1 | 0.99999544 | 0.14062500 | 0.03506000 | False |
| b4_t17 | 0.99999883 | 0.14062500 | 0.00815754 | True |
| b4_t128 | 0.99999888 | 0.18750000 | 0.00679530 | False |
| cached_teacher_clean_vs_fla | 0.99999899 | 0.09375000 | 0.00709437 | True |
| clean_cached_vs_clean_full | 1.00000000 | 0.00000000 | 0.00000000 | True |
| fla_cached_vs_fla_full | 0.99999930 | 0.07812500 | 0.00616372 | True |
