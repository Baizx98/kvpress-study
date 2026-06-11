# 端到端系统性能正式实验方案

本文档用于指导后续在三个同级系统仓库中分别运行端到端 serving 实验，并把结果统一保存回 `kvpress-study`。当前比较对象为 `KVCore`、`vLLM`、`InfiniGen`；模型为 Llama-3.1-8B-Instruct、Qwen3-8B、Mistral-7B-Instruct-v0.3；核心指标为 throughput 和 latency。

## 1. 实验目标

### Problem statement

论文需要证明：在长上下文、长输出、请求长度不确定的 serving workload 下，KVCore 相比 vLLM 和 InfiniGen 能在保持相同模型与相同请求输入的前提下，获得更好的吞吐率和请求延迟。

### Hypothesis

KVCore 的 sparse-aware KV lifecycle 管理可以减少长请求中的 KV 内存压力、无效 attention 计算和不必要的数据搬运，因此在长 prompt、长 decode 和较大 batch 下，应表现为：

- 更高的 output throughput，即 `generated_tokens / wall_time`；
- 更低的 request latency，尤其是 P50/P90/P99 end-to-end latency；
- 更低的 per-token latency，包括 TTFT 和 TPOT；
- 在高并发或长输出设置下，比 dense vLLM 和 InfiniGen 更不容易出现 preemption、OOM 或吞吐崩塌。

### Method

不要让三个系统各自随机采样数据。先在 `kvpress-study` 生成统一 workload manifest，每条请求固定：

- `request_id`
- `source_dataset`
- `prompt`
- `prompt_token_len`
- `target_output_len`
- `max_new_tokens`
- `sampling_config`
- `model_name`
- `tokenizer_path`
- `seed`

然后三个系统分别读取同一份 manifest，运行后只把测量结果写回统一目录。这样可以隔离系统差异，避免数据集、prompt 长度、输出长度、随机采样设置不一致导致结果不可比。

## 2. 仓库、模型和数据集位置

### 系统仓库

当前同级目录已经核实存在：

| system | path |
|---|---|
| KVCore | `/home10T/bzx/workspace/KVCore` |
| vLLM | `/home10T/bzx/workspace/vllm` |
| InfiniGen | `/home10T/bzx/workspace/InfiniGen` |
| result owner | `/home10T/bzx/workspace/kvpress-study` |

所有实验结果只进入 `kvpress-study`，不要分散留在三个系统仓库中。三个系统仓库中可以保留临时日志，但最终 raw metrics、配置和聚合表都要回填到本项目。

### 模型路径

当前 shell 中 `~/Tan/model` 展开为 `/home/bzx/Tan/model`。建议在配置中使用绝对路径，避免不同仓库启动脚本的 home 目录解析不一致。

| model key | local path |
|---|---|
| `llama31_8b_instruct` | `/home/bzx/Tan/model/Llama-3.1-8B-Instruct` |
| `qwen3_8b` | `/home/bzx/Tan/model/Qwen3-8B` |
| `mistral_7b_instruct_v03` | `/home/bzx/Tan/model/Mistral-7B-Instruct-v0.3` |

每个模型单独生成 workload manifest，因为不同 tokenizer 下同一文本的 token 数可能不同。跨系统比较时，必须固定同一个模型对应的 tokenizer 和同一个 manifest。

### 数据集路径

建议把性能实验的数据来源分成三类。

| workload family | purpose | local source |
|---|---|---|
| LongBench selected tasks | 长 prompt、真实 QA/摘要/code 输入 | `/home/bzx/Tan/dataset/LongBench` |
| PG19 continuation | 长上下文、长输出、不容易过早 EOS | `/home/bzx/Tan/dataset/pg19-test/data/test-00000-of-00001-29a571947c0b5ccc.parquet` |
| LongGenBench prompts | 长输出倾向的补充 workload | `/home/bzx/Tan/dataset/LongGenBench/data` |

正式首轮主实验使用 PG19 continuation。PG19 更适合制造长 decode trace；LongBench 和 LongGenBench 暂作为后续补充 workload，不混入当前主实验。

## 3. 统一结果目录

正式实验名建议：

```text
end2end_serving_kvcore_vllm_infinigen_longreq
```

结果目录：

```text
evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/
  README.md
  artifacts/
    configs/
    manifests/
    raw/
      kvcore/
      vllm/
      infinigen/
    logs/
      kvcore/
      vllm/
      infinigen/
    summaries/
    environment/
```

后续画图目录：

```text
figure/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/
```

分析文档目录：

```text
note/end_to_end_system_performance_results_zh.md
```

## 4. Workload manifest 格式

每个模型、每个 workload family、每个 sweep 点生成一份 JSONL manifest。文件名必须包含模型、输入长度、输出长度和 batch/concurrency 设置。

示例路径：

```text
evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/manifests/
  llama31_8b_instruct__pg19__in8k_out2k__bs1.jsonl
  llama31_8b_instruct__pg19__in8k_out2k__bs4.jsonl
  qwen3_8b__longbench__in16k_out1k__bs8.jsonl
```

单条请求建议格式：

```json
{
  "request_id": "pg19_llama31_8b_instruct_in8192_out2048_seed2026_000001",
  "model_key": "llama31_8b_instruct",
  "model_path": "/home/bzx/Tan/model/Llama-3.1-8B-Instruct",
  "tokenizer_path": "/home/bzx/Tan/model/Llama-3.1-8B-Instruct",
  "source_dataset": "pg19",
  "source_id": "test-00000:row123:offset0",
  "prompt": "...",
  "prompt_token_len": 8192,
  "target_output_len": 2048,
  "max_new_tokens": 2048,
  "sampling": {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "ignore_eos": true,
    "seed": 2026
  },
  "workload": {
    "arrival_mode": "closed_loop_batch",
    "batch_size": 4,
    "input_len_bucket": 8192,
    "output_len_bucket": 2048
  }
}
```

关键约束：

- `temperature=0.0` 用于减少随机性；如果某系统不支持完全 greedy，就记录实际 sampling 参数。
- 长输出主实验建议 `ignore_eos=true`，否则不同模型可能提前停止，导致输出长度不可比。若系统不支持 ignore EOS，则必须记录 `actual_output_len`，并单独标注为受 EOS 影响的结果。
- `prompt_token_len` 必须用该模型 tokenizer 计算，不要用字符数或另一个模型的 tokenizer 代替。
- manifest 一旦生成，不要在系统仓库内二次过滤或重排；允许每个系统脚本只读取指定文件。

## 5. Sweep 设计

### 主实验矩阵

当前不再使用 smoke/reduced 数据作为论文证据。正式实验先固定输入长度，只考察长输出长度和 batch size 对端到端性能的影响。

| dimension | values |
|---|---|
| system | `kvcore`, `vllm`, `infinigen` |
| model | `llama31_8b_instruct`, `qwen3_8b`, `mistral_7b_instruct_v03` |
| workload | `pg19` |
| input length | 6K |
| output length | 1K, 2K, 6K |
| batch size | 1, 8, 16, 24 |
| repeats | 1 |
| seed | 2026 |
| GPU | NVIDIA RTX A6000 |

正式 manifest 规模：

| batch size | requests per point | measured batches |
|---|---|---|
| 1 | 32 | 32 |
| 8 | 32 | 4 |
| 16 | 32 | 2 |
| 24 | 48 | 2 |

请求数规则：

```text
requests_per_point = max(32, 2 * batch_size)
```

因此正式 PG19 manifest 数量为：

```text
3 models x 1 workload x 1 input length x 3 output lengths x 4 batch sizes = 36 manifests
```

### 为什么这样设计

- 固定 6K 输入长度，避免单卡 A6000 上 prefill 过重，同时仍保留较长 prompt 的 KV cache 压力。
- 输出长度覆盖 1K/2K/6K，直接对应长输出请求；6K 输出用于放大 decode-stage KV 管理差异。
- batch size 覆盖 1/8/16/24，可以同时观察单请求 latency、常规 batch throughput 和大 batch 下的吞吐/延迟变化。
- 每个点至少 32 条请求；BS=24 时提高到 48 条，保证至少两个完整 measured batches。这样每个点不是 smoke，而是可聚合的正式实验点。
- 当前 repeat 仍为 1，但结果必须显式记录 `repeat_count=1`；如果关键图中某些点波动明显，再对这些点补 repeat。

### 资源风险控制

如果 6K input + 6K output + BS=24 在任一系统上 OOM 或超时，不应删除该点，而是记录为 failed point：

```json
{
  "status": "oom",
  "failure_reason": "CUDA OOM during prefill",
  "max_memory_allocated_gb": 47.3
}
```

当前不主动做压力测试，但如果 BS=24 或 output=6K 出现 OOM/timeout，需要保留 failed point。主图需要明确区分 failed、timeout、completed。

## 6. 运行模式

当前正式实验只执行 closed-loop batch。暂不做压力测试，也暂不做 continuous arrival serving；这样可以先在 A6000 单卡上获得可比较的 output throughput 和 latency。

### Closed-loop batch

含义：一次提交固定 batch size 的请求，所有请求完成后再开始下一批。

用途：

- 控制变量最干净；
- 适合画 throughput/latency 随 input length、output length、batch size 的变化曲线；
- 三个系统最容易统一。

核心指标：

- batch wall time
- per-request end-to-end latency
- TTFT
- TPOT
- output tokens/s
- total tokens/s
- peak GPU memory

## 7. Raw result 格式

每个系统每次运行输出一份 JSONL，每条 request 一行。路径示例：

```text
evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/
  vllm__llama31_8b_instruct__pg19__in8k_out2k__bs4__seed2026.jsonl
```

单条结果建议格式：

```json
{
  "request_id": "pg19_llama31_8b_instruct_in8192_out2048_seed2026_000001",
  "system": "vllm",
  "model_key": "llama31_8b_instruct",
  "workload_family": "pg19",
  "input_len_bucket": 8192,
  "output_len_bucket": 2048,
  "batch_size": 4,
  "repeat_id": 0,
  "repeat_count": 1,
  "seed": 2026,
  "status": "completed",
  "prompt_token_len": 8192,
  "actual_output_len": 2048,
  "submit_time_s": 12.381,
  "first_token_time_s": 14.020,
  "finish_time_s": 45.772,
  "queue_latency_s": 0.0,
  "ttft_s": 1.639,
  "e2e_latency_s": 33.391,
  "tpot_ms": 15.34,
  "itl_mean_ms": 15.31,
  "itl_p50_ms": 14.90,
  "itl_p90_ms": 18.42,
  "itl_p99_ms": 26.51,
  "gpu_peak_memory_gb": 39.8,
  "preemptions": 0,
  "notes": ""
}
```

每次运行还要有一个 run-level metadata JSON：

```json
{
  "system": "vllm",
  "system_repo": "/home10T/bzx/workspace/vllm",
  "system_git_commit": "...",
  "kvpress_study_git_commit": "...",
  "model_key": "llama31_8b_instruct",
  "model_path": "/home/bzx/Tan/model/Llama-3.1-8B-Instruct",
  "manifest_path": "...",
  "gpu_name": "NVIDIA RTX A6000",
  "cuda_visible_devices": "0",
  "driver_version": "...",
  "cuda_version": "...",
  "torch_version": "...",
  "command": "...",
  "start_time": "2026-06-08T...",
  "end_time": "2026-06-08T..."
}
```

## 8. 聚合表格式

每轮实验结束后，在 `artifacts/summaries/` 生成至少两张 CSV。

### `request_metrics.csv`

一行一个请求，直接由 raw JSONL 合并而来。核心列：

```text
system,model_key,workload_family,input_len_bucket,output_len_bucket,batch_size,seed,request_id,status,
repeat_id,repeat_count,prompt_token_len,actual_output_len,ttft_s,e2e_latency_s,tpot_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms,
gpu_peak_memory_gb,preemptions
```

### `aggregate_metrics.csv`

一行一个实验点。核心列：

```text
system,model_key,workload_family,input_len_bucket,output_len_bucket,batch_size,
repeat_count,num_requests,num_completed,num_failed,
throughput_output_toks_per_s,throughput_total_toks_per_s,
latency_mean_s,latency_p50_s,latency_p90_s,latency_p99_s,
ttft_mean_s,ttft_p50_s,ttft_p90_s,ttft_p99_s,
tpot_mean_ms,tpot_p50_ms,tpot_p90_ms,tpot_p99_ms,
gpu_peak_memory_gb_mean,gpu_peak_memory_gb_max,
preemption_rate,oom_rate,timeout_rate
```

吞吐率定义必须固定：

- `throughput_output_toks_per_s = sum(actual_output_len) / measured_wall_time_s`
- `throughput_total_toks_per_s = sum(prompt_token_len + actual_output_len) / measured_wall_time_s`

论文主文建议优先报告 output throughput，因为 decode-heavy 长输出请求的瓶颈主要体现在生成 token 的持续成本；total throughput 可作为补充指标，避免忽略 prefill 代价。

## 9. 使用流程

### Step 1: 在 kvpress-study 生成 manifests

后续应新增一个 manifest 生成脚本，例如：

```text
evaluation/build_end2end_serving_manifests.py
```

建议命令形式：

```bash
cd /home10T/bzx/workspace/kvpress-study

.venv/bin/python evaluation/build_end2end_serving_manifests.py \
  --output-dir evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/manifests_formal_pg19_in6k \
  --models llama31_8b_instruct qwen3_8b mistral_7b_instruct_v03 \
  --workloads pg19 \
  --input-lens 6144 \
  --output-lens 1024 2048 6144 \
  --batch-sizes 1 8 16 24 \
  --num-requests-per-point 32 \
  --measured-batches-per-point 2 \
  --seeds 2026 \
  --repeat-count 1
```

脚本要保证：

- 每个 manifest 内的请求顺序固定；
- 记录 prompt 原文，不依赖运行系统再次访问数据集；
- 记录 tokenizer 计算出的 token 长度；
- 对过短样本做 deterministic 拼接或跳过，并在 metadata 中记录；
- 对超长样本做 tokenizer-aware 截断，而不是字符截断。

### Step 2: 在每个系统仓库运行

每个系统应实现一个适配脚本，读取 manifest 并输出统一 raw result。

建议接口：

```bash
cd /home10T/bzx/workspace/vllm

python run_kvcore_study_manifest.py \
  --manifest /home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/manifests_formal_pg19_in6k/llama31_8b_instruct__pg19__in6k_out1k__bs1__seed2026.jsonl \
  --output /home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/raw/vllm/vllm__llama31_8b_instruct__pg19__in6k_out1k__bs1__seed2026.jsonl \
  --metadata-output /home10T/bzx/workspace/kvpress-study/evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/environment/vllm__llama31_8b_instruct__pg19__in6k_out1k__bs1__seed2026.json \
  --batch-size 1
```

KVCore 和 InfiniGen 同理，只替换系统目录、脚本和 raw 输出目录。

### Step 3: 在 kvpress-study 聚合

后续应新增聚合脚本：

```text
evaluation/summarize_end2end_serving_results.py
```

建议命令：

```bash
cd /home10T/bzx/workspace/kvpress-study

.venv/bin/python evaluation/summarize_end2end_serving_results.py \
  --experiment-dir evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq \
  --output-dir evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/summaries
```

### Step 4: 画论文图

建议主图：

1. Output throughput vs output length，按 system 分线，分 model 或分 panel。
2. P99 e2e latency vs batch size，按 system 分线。
3. Throughput-latency tradeoff，即 x 轴 throughput，y 轴 P99 latency。
4. Completed/failed point table，标出哪些组合在 A6000 上完成、OOM 或 timeout，作为后续收缩 sweep 的依据。

## 10. 公平性控制

必须固定：

- 相同模型权重路径；
- 相同 tokenizer；
- 相同 prompt；
- 相同 `max_new_tokens`；
- 相同 sampling 参数；
- 相同 GPU 型号和数量：当前固定为单卡 NVIDIA RTX A6000；
- 相同测量窗口；
- 相同 warmup 策略；
- 相同 batch size 定义。

必须记录但不强行相同：

- 每个系统的 git commit；
- attention backend；
- block size / page size；
- KV cache dtype；
- tensor parallel 设置；
- 是否支持 prefix cache；
- 是否支持 CUDA graph；
- 是否启用 chunked prefill；
- 是否启用 offload 或 sparse attention。

建议默认关闭会引入额外跨请求复用的优化，例如 prefix cache，除非三个系统都支持并且语义一致。否则主实验应测“不依赖共享前缀”的长请求 serving 能力。

## 11. 关键风险和处理

| risk | impact | handling |
|---|---|---|
| EOS 提前停止 | 输出长度不一致，吞吐不可比 | 主实验启用 `ignore_eos=true`；不支持时单独记录并报告 actual output length |
| 不同系统 batch 定义不同 | batch size 对比失真 | manifest 写入 batch size，同时 raw metadata 记录实际 scheduler 参数 |
| 6K input / 6K output / BS=24 OOM | 大 batch 长输出点缺失 | 保留 failed point，不删除，不用低压点替代 |
| tokenizer 差异 | 输入长度不可比 | 每个模型用本地模型 tokenizer 生成 manifest |
| warmup 不一致 | 首轮延迟偏高 | 每个实验点先 warmup 1-2 batch，不计入 measured window |
| 数据集二次读取 | 三系统拿到不同 prompt | manifest 保存完整 prompt，系统只读取 manifest |
| 系统日志格式不同 | 后处理困难 | 每个系统适配脚本统一写 JSONL schema |

## 12. 正式执行顺序

1. 在 `kvpress-study` 生成 `manifests_formal_pg19_in6k/`。
2. 在 vLLM、KVCore、InfiniGen 中分别跑完整 36 个 manifests。
3. 每个系统输出 raw JSONL 和 metadata JSON，不允许修改 manifest。
4. 回到 `kvpress-study` 聚合三个系统结果，生成 `request_metrics.csv` 和 `aggregate_metrics.csv`。
5. 先画 PG19 fixed-input 主图：output throughput vs output length、P99 latency vs batch size、throughput-latency tradeoff。
6. 如果正式 PG19 结果有清晰趋势，再决定是否补 LongBench 或更长 input length；这些作为后续扩展，不混入当前主实验。

## 13. 当前边界

当前已经实现：

- manifest 生成脚本：`evaluation/build_end2end_serving_manifests.py`
- 正式 PG19 manifests：`evaluation/results/experiments/end2end_serving_kvcore_vllm_infinigen_longreq/artifacts/manifests_formal_pg19_in6k/`

仍待实现：

- 三个系统的 manifest runner；
- raw result 聚合脚本；
- 论文图绘制脚本。

当前版本暂不包含 pressure test 或 continuous arrival。
