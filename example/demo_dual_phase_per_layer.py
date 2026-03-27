from transformers import pipeline

from kvpress import DualPhasePerLayerPress

# 与 demo.py 保持一致：按你的本地模型路径配置
model_path = "/Tan/model"
model_name = "Llama-3.1-8B-Instruct"
model = f"{model_path}/{model_name}"

device = "cuda:1"
pipe = pipeline("kv-press-text-generation", model=model, device=device, dtype="auto")

# 长文本背景 + 短问题，用于测试 prefill 阶段的压缩效果
context = """
北京，中华人民共和国的首都和中央直辖市，是中国的政治、文化、科技和国际交往中心。
作为全球重要的城市，北京拥有三千多年的建城历史和八百多年的建都史。北京在中国历史上
曾多次作为王朝的都城，包括金、元、明、清等朝代，这使得北京积累了丰富的历史文化遗产。

北京的地理位置优越，位于华北平原西北边缘，与天津和河北相邻。城市总面积达16410平方公里，
常住人口超过2000万，是中国人口最多的城市之一。北京的气候属于温带季风气候，四季分明，
夏季炎热多雨，冬季寒冷干燥。

经济方面，北京是中国的经济中心之一，拥有完善的金融体系和众多的国际企业总部。
高新技术产业在北京经济中占有重要地位，包括信息技术、生物技术、新能源等领域。
服务业是北京经济的支柱产业，占GDP的比重超过70%。

文化方面，北京拥有大量的历史文化古迹，包括长城、故宫、颐和园、天坛等世界文化遗产。
北京的教育资源丰富，拥有众多名牌大学，如清华大学、北京大学等，这使得北京成为
中国的科研中心。北京还举办过多次重要的国际活动，包括2008年北京奥运会，展示了
中国的开放和进步。

交通方面，北京拥有完善的公共交通系统，包括地铁、公交、出租车等。首都国际机场
和大兴国际机场是连接中国与世界的重要枢纽。北京的铁路网络发达，是中国的交通枢纽。

未来发展方面，北京正在推进建设国际一流的和谐宜居之都，积极推进京津冀协同发展。
北京还在绿色发展、创新驱动、城市更新等方面进行了大量投资和改革。
"""
question = "北京是什么？"

# 这里演示两点：
# 1) 支持按阶段设置默认压缩率（default_phase_ratios=[prefill, decode]）
# 2) 支持按层覆盖压缩率（layer_phase_ratios）
press = DualPhasePerLayerPress.init_class_vars(
    layer_phase_ratios={},
    default_phase_ratios=[0.99, 0.7],
    block_size=4,
    compression_interval=10000,
)

result = pipe(context, question=question, press=press)
print(result["answer"])
 