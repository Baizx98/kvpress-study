# PG19 dataset

[PG19](https://github.com/google-deepmind/pg19) 是一个面向长上下文语言建模的 benchmark，原始数据来自 Project Gutenberg 的整本书文本。官方常见用法是对长文本 continuation 计算 likelihood / perplexity，而不是做问答式生成准确率评测。

## Project adaptation

本项目为 `PG19` 增加了一条单独的 language-modeling 评测分支，而不是复用 `LongBench` 那类 `context/question/answer_prefix -> generated answer` 的路径。

当前实现的语义是：

- 从 PG19 `test` split 读取整本书文本
- 用模型 tokenizer 切分
- 取前 `max_context_length` 个 token 作为 prefill context
- 取其后的 `pg19_target_tokens` 个 token 作为 continuation target
- 在可选 prefill compression 下计算 target 的负对数似然
- 汇总 `subword_perplexity` 与 `word_perplexity`

这保留了 PG19 “长文本 continuation 建模”的核心语义，但仍不同于官方论文里对整本书或更长窗口做的完整 perplexity 统计。当前实现更适合：

- 评估 prefill compression 是否破坏长书 continuation 建模
- 与 `LongBench` / `needle_in_haystack` 形成互补
- 在研究前期快速验证压缩方法对 language modeling 的影响

## Source datasets

- 默认正式数据源：`pg19`
- 推荐 smoke test 数据源：`emozilla/pg19-test`

`pg19` 在 Hugging Face 上需要 `trust_remote_code=True`，并且其 builder 会从官方公开地址下载元数据与书本文本。若网络环境不稳定，建议先用 `emozilla/pg19-test` 做流程验证。

## Minimal smoke test

可以先用下面这条命令验证本项目里的 PG19 适配链路：

```bash
./.venv/bin/python evaluation/evaluate.py \
  --dataset pg19 \
  --pg19_source_dataset emozilla/pg19-test \
  --model sshleifer/tiny-gpt2 \
  --device cpu \
  --press_name no_press \
  --fraction 0.01 \
  --max_context_length 32 \
  --pg19_target_tokens 8 \
  --output_dir ./evaluation/results/ad_hoc_baselines/pg19_smoke
```
