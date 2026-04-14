# blockwise_stage3_ratio70_fraction20_primarybench 分析

## 完整性

- `longbench:qasper`：5/5
- `longbench:multifieldqa_en`：5/5
- `longbench:hotpotqa`：5/5
- `longbench:2wikimqa`：5/5
- `longbench:musique`：5/5
- `longbench:triviaqa`：5/5
- `needle_in_haystack:16384`：5/5
- `pg19:test`：0/5

## 当前最优

- `longbench:qasper`：blockwise_multi_rep = 40.6100（越高越好）
- `longbench:multifieldqa_en`：blockwise_adaptive_fusion_v1 = 57.6500（越高越好）
- `longbench:hotpotqa`：blockwise_main = 56.2700（越高越好）
- `longbench:2wikimqa`：chunkkv_prefill = 45.1300（越高越好）
- `longbench:musique`：chunkkv_prefill = 35.6300（越高越好）
- `longbench:triviaqa`：blockwise_main = 96.0000（越高越好）
- `needle_in_haystack:16384`：blockwise_multi_rep = 73.4962（越高越好）
- `pg19:test`：暂无结果

## 各数据集结果

### longbench:qasper
- `blockwise_main`：40.31
- `blockwise_multi_rep`：40.61
- `blockwise_adaptive_fusion_v1`：39.48
- `blockwise_multi_rep_diverse_v1`：39.80
- `chunkkv_prefill`：39.70

### longbench:multifieldqa_en
- `blockwise_main`：54.30
- `blockwise_multi_rep`：57.38
- `blockwise_adaptive_fusion_v1`：57.65
- `blockwise_multi_rep_diverse_v1`：53.35
- `chunkkv_prefill`：53.85

### longbench:hotpotqa
- `blockwise_main`：56.27
- `blockwise_multi_rep`：54.37
- `blockwise_adaptive_fusion_v1`：54.37
- `blockwise_multi_rep_diverse_v1`：54.17
- `chunkkv_prefill`：54.61

### longbench:2wikimqa
- `blockwise_main`：40.54
- `blockwise_multi_rep`：42.56
- `blockwise_adaptive_fusion_v1`：38.76
- `blockwise_multi_rep_diverse_v1`：39.81
- `chunkkv_prefill`：45.13

### longbench:musique
- `blockwise_main`：30.45
- `blockwise_multi_rep`：31.90
- `blockwise_adaptive_fusion_v1`：31.90
- `blockwise_multi_rep_diverse_v1`：29.36
- `chunkkv_prefill`：35.63

### longbench:triviaqa
- `blockwise_main`：96.00
- `blockwise_multi_rep`：96.00
- `blockwise_adaptive_fusion_v1`：96.00
- `blockwise_multi_rep_diverse_v1`：93.00
- `chunkkv_prefill`：93.50

### needle_in_haystack:16384
- `blockwise_main`：avg_rouge_l_f=68.21
- `blockwise_multi_rep`：avg_rouge_l_f=73.50
- `blockwise_adaptive_fusion_v1`：avg_rouge_l_f=69.85
- `blockwise_multi_rep_diverse_v1`：avg_rouge_l_f=72.58
- `chunkkv_prefill`：avg_rouge_l_f=70.97

### pg19:test
- 暂无结果

## 最终失败项

- `longbench:qasper__blockwise_adaptive_fusion_v1`: attempts=3, reason=unknown
- `longbench:qasper__blockwise_adaptive_fusion_v1`: attempts=3, reason=unknown
- `longbench:multifieldqa_en__blockwise_adaptive_fusion_v1`: attempts=3, reason=unknown
- `longbench:hotpotqa__blockwise_adaptive_fusion_v1`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_main`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_multi_rep`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_adaptive_fusion_v1`: attempts=3, reason=unknown
- `needle_in_haystack:16384__blockwise_multi_rep_diverse_v1`: attempts=3, reason=unknown
- `needle_in_haystack:16384__chunkkv_prefill`: attempts=3, reason=unknown
- `pg19:test__blockwise_main`: attempts=3, reason=pg19_network
- `pg19:test__blockwise_multi_rep`: attempts=3, reason=network
- `pg19:test__blockwise_adaptive_fusion_v1`: attempts=3, reason=pg19_network
- `pg19:test__blockwise_multi_rep_diverse_v1`: attempts=3, reason=pg19_network
- `pg19:test__chunkkv_prefill`: attempts=3, reason=network
