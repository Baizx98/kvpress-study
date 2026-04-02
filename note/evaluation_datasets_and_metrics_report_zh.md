# KVPress 项目评测数据集与指标报告

## 1. 报告目的

本文档面向本仓库当前评测框架，系统整理 `evaluation/` 下已接入 benchmark 的数据集来源、测试能力、指标计算方式、适用场景与使用注意事项，供后续 KV cache compression、prefill compression、decoding compression、query-aware/block-wise 方法评测时作为统一参照。

本报告基于以下源码整理：

- [`evaluation/evaluate.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate.py)
- [`evaluation/evaluate_registry.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_registry.py)
- [`evaluation/evaluate_config.yaml`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_config.yaml)
- `evaluation/benchmarks/*/README.md`
- `evaluation/benchmarks/*/calculate_metrics.py`
- `evaluation/benchmarks/*/create_huggingface_dataset.py`

说明：

- 本报告描述的是“本项目中的实际评测方式”，不完全等同于各原始 benchmark 官方论文或官方脚本。
- 对于一些非平稳或强依赖外部平台的 benchmark，本报告会明确标出限制。
- 由于用户后续主要做 KV 压缩研究，报告会偏向“实验选型”和“指标解释”，而不是纯 benchmark 科普。

---

## 2. 本项目评测框架总览

### 2.1 评测入口

本项目统一通过 [`evaluation/evaluate.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate.py) 执行评测，核心流程为：

1. 解析配置并实例化 `EvaluationConfig`
2. 根据 `PRESS_REGISTRY` 初始化压缩方法
3. 根据 `DATASET_REGISTRY` 从 Hugging Face 加载测试集
4. 必要时对数据做二次处理
5. 调用 `kv-press-text-generation` pipeline 执行推理
6. 保存 `predictions.csv`
7. 调用 `SCORER_REGISTRY` 里的 scorer 计算指标并保存 `metrics.json`
8. 保存本次实验 `config.yaml`

### 2.2 当前注册的数据集

见 [`evaluation/evaluate_registry.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_registry.py)：

| 数据集键名 | Hugging Face 数据源 | 说明 |
| --- | --- | --- |
| `loogle` | `simonjegou/loogle` | 长依赖理解与长文生成 |
| `ruler` | `simonjegou/ruler` | 合成长上下文任务集 |
| `zero_scrolls` | `simonjegou/zero_scrolls` | 长文自然语言任务集合 |
| `infinitebench` | `MaxJeblick/InfiniteBench` | 多能力长上下文 benchmark |
| `longbench` | `Xnhyacinth/LongBench` | 真实长文本多任务评测 |
| `longbench-e` | `Xnhyacinth/LongBench` | 按长度分桶的 LongBench 版本 |
| `longbench-v2` | `simonjegou/LongBench-v2` | 选择题式长上下文评测 |
| `needle_in_haystack` | `alessiodevoto/paul_graham_essays` | 长文检索能力 |
| `aime25` | `alessiodevoto/aime25` | AIME 2025 数学推理 |
| `math500` | `alessiodevoto/math500` | MATH-500 数学推理 |

### 2.3 通用输入字段

大多数 benchmark 在本仓库中会被整理成统一字段：

- `context`: 作为长上下文输入的主体内容
- `question`: 问题或指令
- `answer_prefix`: 引导模型输出格式的前缀
- `answer` 或 `answers`: 参考答案
- `task`: 子任务名
- `max_new_tokens`: 默认生成长度

这意味着你后续比较不同压缩方法时，模型入口是统一的，但不同 benchmark 的评分逻辑完全不同，不能直接将分数横向平均解释为“统一总分”。

### 2.4 推理模式

#### 普通压缩模式

当 press 不是 `DecodingPress` 时，评测代码会按相同 `context` 分组，把多个 `question` 一次性喂给同一个上下文。这更符合 prefill 压缩的使用场景，因为同一个长 context 只压缩一次，然后服务多个 query。

#### 解码压缩模式

当 press 属于 `DecodingPress` 时，评测按单条 `(context, question)` 执行。这更适合研究 decode 阶段的 KV 管理，但吞吐模式和上面的 grouped evaluation 不同。

### 2.5 `query_aware` 的真实语义

本项目里 `query_aware=True` 的含义不是“模型知道 query”，而是：

- 在压缩前先把 `question` 直接拼接到 `context` 末尾
- 然后令 `question=""`

因此它本质是“问题感知压缩”，不是标准的 context/question 双塔或分离式输入。

这个行为会直接改变压缩对象，所以：

- 比较 `query_aware=False` 和 `True` 时，输入分布已不同
- 某些 press 会被强制切换成 `query_aware=True`
- 做公平对比时要单独报告这一点

### 2.6 可复现性设置

`evaluate.py` 中显式设置了：

- `torch.manual_seed(seed)`
- `np.random.seed(seed)`
- `random.seed(seed)`
- CUDA deterministic

默认种子来自 [`evaluation/evaluate_config.yaml`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_config.yaml)，目前为 `42`。  
这对采样子集评测和 benchmark 间的可比较性是有帮助的，但仍需注意：

- 生成式模型在某些底层实现上仍可能存在轻微非确定性
- 不同注意力实现、不同显卡和不同量化配置也可能带来微小偏差

---

## 3. 数据集与指标全景建议

下面先给一个面向 KV 压缩研究的快速选型表。

| 数据集 | 主要能力 | 指标类型 | 更适合的研究问题 | 主要限制 |
| --- | --- | --- | --- | --- |
| LooGLE | 长依赖理解、长文摘要、槽位填空 | BLEU/ROUGE/METEOR/BERTScore/匹配率 | 看语义保持和生成质量退化 | 自动指标较多，解释需谨慎 |
| RULER | 检索、多跳、聚合、变量跟踪 | string match | 看超长上下文鲁棒性和长度扩展 | 合成任务占主导 |
| Zero Scrolls | 长文自然任务 | 本仓库内无自动分数 | 生成结果导出、准备外部提交 | 不能在仓库内直接比较最终质量 |
| InfiniteBench | 检索、代码、数学、对话、QA | 任务特定规则分数/F1 | 看能力维度细粒度退化 | 子任务异构，平均值解释需谨慎 |
| LongBench | QA、摘要、分类、检索、计数、代码 | F1/ROUGE/分类/检索/代码相似 | 做论文主结果、综合任务评测 | 任务异构，不宜混成单一总分 |
| LongBench-E | 同 LongBench，但看长度分桶 | 分长度区间分数 | 分析长度增长下的退化趋势 | 长度桶较粗 |
| LongBench-v2 | 选择题长上下文理解 | 准确率 | 看难度/长度分组稳定性 | 强依赖输出格式 |
| Needle in a Haystack | 精确检索 | ROUGE | 看压缩是否保留远距离关键片段 | 偏检索，不能代表复杂推理 |
| AIME25 | 高难数学推理 | boxed accuracy | 看复杂推理对压缩的敏感性 | 只看最终 boxed 答案 |
| MATH500 | 通用数学推理 | boxed accuracy | 看推理型任务的鲁棒性 | 同样格式依赖较强 |

---

## 4. 各数据集详细说明

## 4.1 LooGLE

### 数据集简介

LooGLE 来自长依赖理解方向，README 中说明其目标是评估模型对 short dependency 与 long dependency 内容的理解能力。本仓库实际接入了 4 类任务：

- `shortdep_qa`
- `longdep_qa`
- `shortdep_cloze`
- `longdep_summarization`

数据由 [`evaluation/benchmarks/loogle/create_huggingface_dataset.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/loogle/create_huggingface_dataset.py) 转换为统一格式。

### 测试能力

- 短依赖问答
- 长依赖问答
- 长文摘要
- 基于长文本的实体槽位恢复

它特别适合观察压缩后：

- 细节事实是否仍可被提取
- 跨远距离信息的整合是否下降
- 生成类任务语义质量是否明显退化

### 本项目中的输入组织

- `context` 会带上任务型 instruction prompt
- `question` 对 QA/cloze 任务单独保留
- `answer_prefix` 用于控制输出风格
- `longdep_summarization` 中 `question=""`

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/loogle/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/loogle/calculate_metrics.py)。

#### `shortdep_cloze`

使用两类匹配指标：

- `exact_match`: 预测字典中 key-value 与参考答案完全一致的比例
- `partial_match`: 对同一个 mask，若预测文本与真值文本 token 集有交集则算部分命中

这两个指标更适合测“关键信息是否找回”，比纯生成指标更接近检索/结构化恢复。

#### 其他任务

使用：

- BLEU-1
- BLEU-4
- ROUGE-1 / ROUGE-2 / ROUGE-L
- METEOR
- BERTScore

其中：

- BLEU 更偏 n-gram 精确匹配
- ROUGE 更偏覆盖与召回
- METEOR 对词形和语义变体更宽容
- BERTScore 更偏语义相似度

### 适用场景

- 研究压缩后生成质量变化
- 比较不同压缩率下长文摘要质量
- 观察 query-aware 压缩是否更利于定位问答线索
- 做“检索保留 + 生成保真”的联合分析

### 不足与风险

- 自动文本指标未必与人工质量完全一致
- `BERTScore` 计算成本更高
- `shortdep_cloze` 对输出 JSON 格式有要求，输出格式差会直接伤害分数

### 对 KV 压缩研究的建议

如果你要证明“压缩后不仅能找对信息，还能生成自然答案”，LooGLE 是很好的补充 benchmark，但不建议单独拿它充当主结果，因为其指标较多、解释空间也更大。

---

## 4.2 RULER

### 数据集简介

RULER 是典型的长上下文合成 benchmark，支持不同 context length 配置。本仓库通过 [`evaluation/benchmarks/ruler/create_huggingface_dataset.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/ruler/create_huggingface_dataset.py) 从不同长度目录中读取 JSONL，并拆出：

- `context`
- `question`
- `answer_prefix`
- `answer`
- `task`

README 中提到其覆盖 13 个任务，属于 4 大类能力：

- needle in the haystack
- question answering
- multi-hop tracing
- aggregation

### 测试能力

从脚本中的模式名看，本仓库主要围绕以下类别：

- `niah`: 近似 needle 检索
- `qa`: 问答
- `vt`: 变量追踪
- `cwe`: 高频词/聚合
- `fwe`: 词级别查找/聚合类

它非常适合衡量：

- 超长上下文下关键信息定位能力
- 多步骤符号/变量追踪能力
- 压缩后全局聚合任务是否掉点

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/ruler/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/ruler/calculate_metrics.py)。

核心指标是 `string_match`，但有两种版本：

#### `string_match_part`

用于 `qa_*` 类任务。  
对每个样本，只要参考答案列表中的任意一个字符串出现在预测中，就记为 1，否则为 0。最后求平均并乘 100。

可近似写为：

`score = 100 * mean( max_{r in refs_i}[r in pred_i] )`

#### `string_match_all`

用于非 `qa` 类任务。  
对每个样本，参考答案列表里命中的字符串比例作为该样本得分，然后再对全体样本求平均并乘 100。

可近似写为：

`score = 100 * mean( (#matched_refs_i / #refs_i) )`

### 适用场景

- 研究不同 context length 下压缩鲁棒性
- 快速看压缩是否破坏合成长文检索和聚合能力
- 做超长上下文主结果或长度扩展实验

### 不足与风险

- 是合成任务，不完全等价于真实业务文本
- string match 对语义近似回答不宽容
- README 明确提示：本仓库使用固定 tokenizer 生成 RULER 数据，结果未必可与原论文直接对齐

### 对 KV 压缩研究的建议

RULER 很适合作为“长度扩展性”和“远距离信息保留”主 benchmark，尤其适合配合不同压缩率和不同 context length 画退化曲线。

---

## 4.3 Zero Scrolls

### 数据集简介

Zero Scrolls 是长文自然语言任务集合，README 中提到覆盖 10 个任务，包括：

- summarization
- question answering
- aggregated sentiment classification
- information reordering

本仓库通过 [`evaluation/benchmarks/zero_scrolls/create_huggingface_dataset.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/zero_scrolls/create_huggingface_dataset.py) 将原始 `input` 按索引切分为：

- `context`
- `question`
- `answer_prefix`

任务包括：

- `gov_report`
- `summ_screen_fd`
- `qmsum`
- `qasper`
- `narrative_qa`
- `quality`
- `musique`
- `squality`
- `space_digest`
- `book_sum_sort`

### 测试能力

它覆盖：

- 长文摘要
- 文档问答
- 多文档理解
- 排序/重排
- 分类类任务

### 指标计算方式

当前项目中评分脚本 [`evaluation/benchmarks/zero_scrolls/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/zero_scrolls/calculate_metrics.py) 直接返回空字典：

```python
def calculate_metrics(df):
    return {}
```

README 也明确说明：

- 数据集里不提供最终答案
- 需要把预测提交到 Zero Scrolls 官方网站获取结果

### 适用场景

- 生成预测结果，准备外部官方评测
- 作为长文真实任务数据源做定性分析
- 对某些任务做人工抽样阅读

### 不足与风险

- 仓库内无法直接得到最终质量分数
- 不适合做日常 sweep 或大量 ablation，因为反馈链路较长
- 不适合当前阶段充当主 benchmark

### 对 KV 压缩研究的建议

可以把 Zero Scrolls 作为“对外展示”或“补充 benchmark”，但不适合当你当前 block-wise / prefill 压缩实验的主评测集合。

---

## 4.4 InfiniteBench

### 数据集简介

InfiniteBench 是多能力长上下文 benchmark。其构造脚本 [`evaluation/benchmarks/infinite_bench/create_huggingface_dataset.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/infinite_bench/create_huggingface_dataset.py) 中列出了任务、样本量和平均 token 长度，覆盖很广。

本仓库接入的任务包括：

- `passkey`
- `kv_retrieval`
- `number_string`
- `longdialogue_qa_eng`
- `longbook_qa_eng`
- `longbook_choice_eng`
- `code_run`
- `code_debug`
- `math_find`
- `math_calc`
- `longbook_sum_eng`
- `longbook_qa_chn`

但评分脚本当前不支持 `longbook_sum_eng`，这一点需要单独注意。

### 测试能力

InfiniteBench 的优势在于“能力切片细”：

- 精确检索：`passkey`, `kv_retrieval`, `number_string`
- 代码执行/调试：`code_run`, `code_debug`
- 数学查找/计算：`math_find`, `math_calc`
- 长文 QA：`longbook_qa_eng`, `longbook_qa_chn`
- 长对话人物识别：`longdialogue_qa_eng`
- 长文选择题：`longbook_choice_eng`

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/infinite_bench/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/infinite_bench/calculate_metrics.py)。

总体上，`calculate_metrics(df)` 会读取当前 `task`，再调用 `get_score(labels, preds, task, model)` 对样本逐个打分并取平均。

#### 1. 检索类任务

- `kv_retrieval`: 预测中分词后只要包含目标 value 即判对
- `passkey`: 提取预测中的第一个整数，与 gold 比较
- `number_string`: 同样提取第一个整数串比较

这类评分几乎都是 exact/near-exact matching，特别适合检验压缩后关键 token 是否还在。

#### 2. 代码类任务

- `code_run`: 取预测最后一个可解析整数，与正确返回值比
- `code_debug`: 提取最后一个 A-J 选项、或解析 answer prefix，匹配正确选项/函数名

这类任务适合测压缩对程序理解链路的影响。

#### 3. 数学类任务

- `math_find`: 提取首个整数或浮点，与目标数比较
- `math_calc`: 读取预测中的数字序列，按前缀连续正确长度计分  
  即如果前几个中间结果连续正确，则按 `correct_prefix_len / total_len` 计分

`math_calc` 很有价值，因为它不是单一 final answer EM，而是部分过程正确也能反映出来。

#### 4. QA 类任务

- `longbook_qa_eng`: 英文 QA F1
- `longbook_qa_chn`: 中文 QA F1
- `longdialogue_qa_eng`: 只要预测中包含任意一个正确实体名称就算对
- `longbook_choice_eng`: 从输出中解析 A/B/C/D，命中即对

英文 QA F1 会做 normalize：

- 小写化
- 去标点
- 去冠词
- 规范空格

中文 QA F1 会做：

- 小写化
- 去中英文标点
- 去空白
- 按字符粒度做 F1

### 适用场景

- 想看不同能力是否受压缩影响不同
- 想知道某种 press 是否“检索强但推理弱”
- 想做更细粒度 ablation，而不是只看 LongBench 总体退化

### 不足与风险

- 子任务异构严重
- 把所有任务简单平均后可解释性较弱
- `longbook_sum_eng` 在当前 scorer 中明确不支持，会触发断言

### 对 KV 压缩研究的建议

InfiniteBench 非常适合作为能力画像补充集。若你的 block-wise 方法宣称“更保留结构信息或关键 token”，InfiniteBench 往往能比 LongBench 更快看出差异来源。

---

## 4.5 LongBench

### 数据集简介

LongBench 是长文本评测里非常常用的一组真实任务 benchmark。本仓库通过 [`evaluation/benchmarks/longbench/create_huggingface_dataset.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/longbench/create_huggingface_dataset.py) 接入，并为不同任务补充 instruction prompt 与 answer prefix。

接入的任务包括：

- QA 类：`narrativeqa`, `qasper`, `multifieldqa_en`, `multifieldqa_zh`, `hotpotqa`, `2wikimqa`, `musique`, `triviaqa`
- 摘要类：`gov_report`, `qmsum`, `multi_news`, `vcsum`, `samsum`
- 分类类：`trec`, `lsht`
- 检索/计数类：`passage_retrieval_en`, `passage_retrieval_zh`, `passage_count`
- 代码类：`lcc`, `repobench-p`

### 测试能力

LongBench 的价值在于覆盖面广，而且更偏真实任务：

- 单文档/多文档问答
- 多跳问答
- 会议/政府报告/多新闻摘要
- 分类
- 从长文中找段落
- 统计计数
- 代码补全

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/longbench/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/longbench/calculate_metrics.py)。

对每个样本：

- 如果有多个参考答案，取最佳分
- 某些任务会先截取预测第一行，避免模型输出解释性废话影响评分
- 最终分数通常乘以 100

各任务到指标的映射如下。

#### QA F1

用于：

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `2wikimqa`
- `musique`
- `triviaqa`

做法是标准 token-level F1：

- 英文先 normalize
- 再按 token 计算 precision / recall / F1

#### 中文 QA F1

用于：

- `multifieldqa_zh`

会先用 `jieba` 分词，再做中文规范化和 F1 计算。

#### ROUGE

用于：

- `gov_report`
- `qmsum`
- `multi_news`
- `samsum`

实现中取 `ROUGE-L F`

#### 中文 ROUGE

用于：

- `dureader`
- `vcsum`

先用 `jieba` 分词，再送入 ROUGE。

#### 分类分数

用于：

- `trec`
- `lsht`

逻辑是：

- 找出预测中出现的类别名
- 若预测中包含多个类别名且其中有歧义，会做去歧义处理
- 若 gold 类别在匹配集合中，则得分为 `1 / 匹配类别数`

这意味着模型若输出多个候选类别，会被惩罚。

#### 检索分数

用于：

- `passage_retrieval_en`
- `passage_retrieval_zh`

从预测中抽取数字，与目标段落编号比较。若输出多个数字，则得分为：

`正确数字出现次数 / 输出中数字总数`

这鼓励模型只给单一答案。

#### 计数分数

用于：

- `passage_count`

从输出中抽取所有数字，与 gold 相等的数字占比作为分数。

#### 代码相似度

用于：

- `lcc`
- `repobench-p`

使用 `fuzzywuzzy.fuzz.ratio(prediction, ground_truth) / 100`。  
脚本会优先取预测里第一条看起来像代码而不是注释的行。

### 适用场景

- 论文主结果
- 做跨任务综合评测
- 比较不同压缩方法在真实任务上的泛化
- 辅助判断某方法是更偏“检索型保真”还是“生成型保真”

### 不足与风险

- 各任务指标异构
- ROUGE 与 F1 的数值范围虽然都可以归一到 0-100，但语义不同
- 代码相似度与真实可执行性并不等价

### 对 KV 压缩研究的建议

如果你只能选一个“综合主 benchmark”，LongBench 通常是最合适的。但论文里最好分任务族报告，而不要把所有子任务粗暴平均成一个唯一数字。

---

## 4.6 LongBench-E

### 数据集简介

`longbench-e` 仍使用 LongBench 的 scorer，但更强调长度分桶表现。  
配置入口在 [`evaluation/evaluate_registry.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_registry.py) 中注册为：

- 数据源：`Xnhyacinth/LongBench`
- scorer：`calculate_metrics_e`

### 测试能力

和 LongBench 类似，但核心目的是观察长度变化带来的退化。

### 指标计算方式

见 `calculate_metrics_e(df)`：

- 先按原任务类型计算单样本分数
- 再按照样本 `length` 分入三个桶
- `0-4k`
- `4-8k`
- `8k+`
- 每个桶内求平均，再乘以 100

### 适用场景

- 比较不同压缩方法随长度增长的稳定性
- 证明某种压缩策略在长上下文增长时掉点更慢
- 做长度敏感性分析图

### 不足与风险

- 长度分桶比较粗
- 若样本在高长度桶数量不均衡，结论要结合样本数看

### 对 KV 压缩研究的建议

如果你的方法核心 claim 是“对超长 context 更友好”，LongBench-E 应该和 RULER 一起使用，一个偏真实任务，一个偏合成可控任务，互补性很好。

---

## 4.7 LongBench-v2

### 数据集简介

本仓库通过 [`evaluation/benchmarks/longbenchv2/create_huggingface_dataset.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/longbenchv2/create_huggingface_dataset.py) 将其转成 0-shot 选择题格式。模型会看到：

- 一段长文本
- 一个问题
- 4 个候选项
- 明确要求输出：`The correct answer is (...)`

### 测试能力

- 长上下文阅读理解
- 选择题判断
- 在长度和难度分组上的稳定性

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/longbenchv2/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/longbenchv2/calculate_metrics.py)。

对每条样本：

- 若预测中包含 `The correct answer is ({expected_answer})`
- 或包含 `The correct answer is {expected_answer}`

则判为正确。

最后返回：

- `average`
- 按 `difficulty` 分组的均值
- 按 `length` 分组的均值

### 适用场景

- 需要一个结构简单、解释直接的 benchmark
- 想看不同难度和长度上的分类准确率
- 想快速做大规模 sweep

### 不足与风险

- 对输出格式较敏感
- 即便模型“知道答案”，如果没按模板写也会记错

### 对 KV 压缩研究的建议

LongBench-v2 可以作为“轻量、稳定、便于做趋势图”的补充 benchmark，但不建议单独作为主实验，因为它主要反映选择题准确率。

---

## 4.8 Needle in a Haystack

### 数据集简介

README 指出该 benchmark 用于测试模型在大段文本中检索隐藏 needle 的能力。本仓库以 `paul_graham_essays` 为 haystack 数据源，并在运行时动态插入 needle。

特殊处理逻辑位于：

- [`evaluation/evaluate.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate.py)
- [`evaluation/benchmarks/needle_in_haystack/utils.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/needle_in_haystack/utils.py)

### 测试能力

- 远距离关键信息检索
- 压缩后目标句是否仍被保留
- 不同插入深度下的鲁棒性

### 本项目中的特殊要求

若数据集为 `needle_in_haystack`，配置中必须提供：

- `needle_depth`
- `max_context_length`

否则 `EvaluationConfig.__post_init__()` 会直接报错。

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/needle_in_haystack/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/needle_in_haystack/calculate_metrics.py)。

每个样本对：

- `needle`
- `predicted_answer`

计算一组 ROUGE 分数，返回的是每条样本的 ROUGE 结果列表，而不是单个平均值。

因此 `metrics.json` 结构会与其他 benchmark 不同。

### 适用场景

- block-wise 压缩是否更能保留远距离细粒度信息
- 比较不同压缩率、不同插入深度下的检索退化
- 分析 query-aware 对检索帮助是否明显

### 不足与风险

- 主要测 retrieval，不测复杂 reasoning
- 返回的是 per-example ROUGE 列表，后处理时你可能还要自己再聚合

### 对 KV 压缩研究的建议

这是非常适合 prefill compression 的定向测试集。若你的方法强调“不会丢掉远距离少量关键 token”，Needle in a Haystack 应该作为专门的 supporting experiment。

---

## 4.9 AIME25

### 数据集简介

AIME25 来自 2025 年 AIME-I 和 AIME-II 数学竞赛题，属于高难度数学推理集合。

### 测试能力

- 多步数学推理
- 高精度 final answer 生成
- 压缩对复杂 reasoning 链的影响

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/aime25/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/aime25/calculate_metrics.py)。

逻辑非常直接：

1. 从预测中抽取 `boxed{...}` 内的内容
2. 与标准答案做字符串比较
3. 统计：
   - `correct`
   - `answered`：是否出现了 `boxed{`
   - `accuracy = correct / total`
   - `total`

### 适用场景

- 检验压缩对高难推理的最终答案影响
- 观察 decoding compression 是否破坏推理质量

### 不足与风险

- 只看最终 boxed 答案，不看中间过程
- 如果模型答案对了但没用 `boxed{}`，会被记作未答或错误

### 对 KV 压缩研究的建议

若你需要说明“压缩不只影响检索，也会影响深度推理”，AIME25 很有说服力。但因为格式依赖较强，最好配合 few-shot prompt 规范输出，或额外记录 raw answer 进行人工 spot check。

---

## 4.10 MATH500

### 数据集简介

MATH500 由 MATH benchmark 的一个 500 题子集构成，README 中说明来源与 OpenAI 的 `Let's Verify Step by Step` 工作一致。

### 测试能力

- 通用数学推理
- 比 AIME 更宽泛的数学题型覆盖

### 指标计算方式

评分脚本见 [`evaluation/benchmarks/math500/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/math500/calculate_metrics.py)。

当前实现与 AIME25 基本相同：

- 提取 `boxed{...}`
- 与 gold 字符串比较
- 输出 `correct`, `answered`, `accuracy`, `total`

### 适用场景

- 比较不同压缩方法在一般数学推理上的稳定性
- 与 AIME25 形成“竞赛题 + 通用数学题”的组合

### 不足与风险

- 同样强依赖 boxed 输出格式
- 无法反映过程质量，只反映 final answer

### 对 KV 压缩研究的建议

MATH500 和 AIME25 可以成对使用。若两者都掉点，通常说明压缩已开始影响深层 reasoning；若只在 AIME 掉点更明显，则可能是高难题对上下文噪声更敏感。

---

## 5. 面向 KV 压缩研究的测试选型建议

## 5.1 如果你的重点是 Prefill Compression

建议优先组合：

- `RULER`
- `LongBench`
- `Needle in a Haystack`

理由：

- `RULER`：看超长长度与合成控制任务退化
- `LongBench`：看真实任务质量
- `Needle in a Haystack`：看关键远距离 token 保留能力

## 5.2 如果你的重点是 Decoding Compression

建议优先组合：

- `AIME25`
- `MATH500`
- `InfiniteBench`

理由：

- 数学题和部分代码/推理任务对 decode 阶段错误更敏感
- `DecodingPress` 在 `evaluate.py` 中逐样本推理，更贴近这类任务模式

## 5.3 如果你的重点是 Query-aware / 问题感知压缩

建议优先组合：

- `Needle in a Haystack`
- `RULER`
- `InfiniteBench` 中的检索类任务

因为这些任务最容易直接观察“把问题拼进 context 再压缩”是否提升目标信息保留。

## 5.4 如果你的重点是论文主结果

建议主结果至少包含：

- `LongBench`
- `RULER`

再增加一个定向补充：

- 检索导向：`Needle in a Haystack`
- 能力画像导向：`InfiniteBench`
- 推理导向：`AIME25 + MATH500`

## 5.5 如果你的重点是快速回归测试

建议：

- 使用 `fraction < 1.0`
- 先跑 `LongBench-v2` 或 `RULER`
- 再挑一两个 LongBench 子任务做 smoke test

这样反馈更快，适合频繁改 block-wise 逻辑后的 sanity check。

---

## 6. 指标解释与汇报建议

对 KV 压缩论文或实验报告，建议至少同时汇报三类指标：

### 6.1 任务质量

来自 benchmark 本身：

- Accuracy
- F1
- ROUGE
- string match
- exact/partial match

### 6.2 系统性能

建议额外记录：

- latency
- throughput
- peak memory
- 实际压缩后 KV 大小

因为只报告任务指标不能体现压缩方法的系统价值。

### 6.3 代价收益

建议至少给出：

- 质量下降 vs 显存节省
- 质量下降 vs 吞吐提升
- 质量下降 vs 压缩率

对于系统论文，这类 trade-off 曲线通常比单点分数更重要。

---

## 7. 当前框架中的几个重要坑

### 7.1 `zero_scrolls` 不能在仓库内直接出最终分

当前 scorer 返回空字典，只能导出预测后再走外部官方平台。

### 7.2 `needle_in_haystack` 返回结构与其他 benchmark 不同

它返回的是每个样本的 ROUGE 列表，不是单个平均数。后续统计时要自己再聚合。

### 7.3 `InfiniteBench` 的 `longbook_sum_eng` 当前不支持评分

评分函数里明确 `raise AssertionError`，不要把这个任务直接纳入自动汇总。

### 7.4 `query_aware=True` 会改变输入分布

这不是简单开关，而是直接把问题拼进 context。做公平比较时要单独注明。

### 7.5 不同 benchmark 的 `metrics.json` 结构不统一

例如：

- LongBench 返回单个数字
- LongBench-E 返回长度桶字典
- LooGLE 返回按 task 分组的多指标字典
- Needle 返回列表

后续如果要做统一汇总脚本，必须按 benchmark 单独适配。

---

## 8. 结论

从当前仓库接入方式看：

- `LongBench` 最适合做综合主结果
- `RULER` 最适合做长上下文长度与结构化能力分析
- `Needle in a Haystack` 最适合做远距离关键信息保留分析
- `InfiniteBench` 最适合做能力画像和 error analysis
- `AIME25` / `MATH500` 最适合做推理敏感性验证
- `Zero Scrolls` 更适合外部提交或补充展示，不适合日常自动评测主线

如果你下一步是继续做 KV 压缩实验，我建议主线先固定为：

1. `LongBench`
2. `RULER`
3. `Needle in a Haystack`

然后按你的 claim 再加：

- 强调推理：`AIME25` + `MATH500`
- 强调能力画像：`InfiniteBench`

这样你的评测体系会比较完整：既有真实任务，也有合成长上下文，还能覆盖检索、推理和系统 trade-off。

---

## 9. 附：关键源码位置索引

- 评测主流程：[`evaluation/evaluate.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate.py)
- 数据集/评分注册：[`evaluation/evaluate_registry.py`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_registry.py)
- 默认配置：[`evaluation/evaluate_config.yaml`](/home10T/bzx/workspace/kvpress-study/evaluation/evaluate_config.yaml)
- LooGLE scorer：[`evaluation/benchmarks/loogle/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/loogle/calculate_metrics.py)
- RULER scorer：[`evaluation/benchmarks/ruler/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/ruler/calculate_metrics.py)
- InfiniteBench scorer：[`evaluation/benchmarks/infinite_bench/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/infinite_bench/calculate_metrics.py)
- LongBench scorer：[`evaluation/benchmarks/longbench/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/longbench/calculate_metrics.py)
- LongBench-v2 scorer：[`evaluation/benchmarks/longbenchv2/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/longbenchv2/calculate_metrics.py)
- Needle scorer：[`evaluation/benchmarks/needle_in_haystack/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/needle_in_haystack/calculate_metrics.py)
- AIME25 scorer：[`evaluation/benchmarks/aime25/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/aime25/calculate_metrics.py)
- MATH500 scorer：[`evaluation/benchmarks/math500/calculate_metrics.py`](/home10T/bzx/workspace/kvpress-study/evaluation/benchmarks/math500/calculate_metrics.py)
