# -*- coding: utf-8 -*-
# Llama-2 本地推理示例（已适配：读取 .env token + Apple Silicon MPS/CPU）
#
# ⚠️ 运行前必须先完成 HuggingFace 授权（否则仍会 403）：
#   1. 注册/登录 HuggingFace 账号
#   2. 打开 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#   3. 点击 "Agree and access repository"，填写并同意 Meta Llama 2 社区许可
#   4. 在 Settings → Access Tokens 创建 Read 权限的 token
#   5. 把该 token 填进本目录 .env 的 HUGGING_FACE_TOKEN=<token>
#   完成后 403 才会消失。若仍 403，说明账号还没通过授权或被拒。

import os
from dotenv import load_dotenv        # 加载 .env
load_dotenv()

HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if not HF_TOKEN:
    raise SystemExit("未在 .env 中找到 HUGGING_FACE_TOKEN，请先配置并完成 Llama-2 授权。")

# 让 transformers / huggingface_hub 也自动带上该 token
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACEHUB_TOKEN"] = HF_TOKEN

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-2-7b-chat-hf"

# 选择可用设备：Apple Silicon(MPS) > CUDA > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备: {device}")

# 加载预训练模型的分词器（必须带 token，否则 gated 模型会 403）
tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)

# 加载预训练的模型：小内存加载 + 半精度，适配本机内存
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    token=HF_TOKEN,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

# 定义一个提示，希望模型基于此提示生成故事
prompt = "请给我讲个玫瑰的爱情故事?"

# 使用分词器将提示转化为模型可以理解的格式，并放到模型所在设备上
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 使用模型生成文本，设置最大生成令牌数为2000
# Llama-2 没有 pad_token，显式指定为 eos_token 以避免告警
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=2000,
    pad_token_id=tokenizer.eos_token_id,
)

# 将生成的令牌解码成文本，并跳过任何特殊的令牌，例如[CLS], [SEP]等
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的响应
print(response)

# ---------------------------------------------------------------------------
# 备选方案（不想授权 / 机器跑不动 7B 时）：把上面 MODEL 换成免授权的小模型即可免 403
#   MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3GB，本机能跑
#   tokenizer / model 的 from_pretrained 请将 token=HF_TOKEN 去掉
# ---------------------------------------------------------------------------