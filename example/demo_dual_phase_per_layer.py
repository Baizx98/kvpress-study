from transformers import pipeline

from kvpress import DualPhasePerLayerPress

# 与 demo.py 保持一致：按你的本地模型路径配置
model_path = "/Tan/model"
model_name = "Llama-3.1-8B-Instruct"
model = f"{model_path}/{model_name}"

device = "cuda:0"
pipe = pipeline("kv-press-text-generation", model=model, device=device, dtype="auto")

# 最小示例上下文
context = "北京是中国的首都，也是全国政治、文化和国际交往中心。"
question = "北京是什么类型的城市？"

# 这里演示两点：
# 1) 支持按阶段设置默认压缩率（default_phase_ratios=[prefill, decode]）
# 2) 支持按层覆盖压缩率（layer_phase_ratios）
press = DualPhasePerLayerPress.init_class_vars(
    layer_phase_ratios={
        8: [0.3, 0.3],  # 第0层：prefill压缩更激进，decode较保守
        7: [0.5, 0.2],  # 第1层：阶段压缩率不同
    },
    default_phase_ratios=[0.2, 0.2],
    block_size=16,
    compression_interval=32,
)

result = pipe(context, question=question, press=press)
print(result["answer"])
