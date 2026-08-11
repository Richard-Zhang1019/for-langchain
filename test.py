# Please install OpenAI SDK first: `pip3 install openai`
import os
import sys
from openai import OpenAI

# 支持从 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 中读取密钥
api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY')
if not api_key:
    print(
        "Missing API key. Please set DEEPSEEK_API_KEY (or OPENAI_API_KEY) and retry.\n"
        "Example: export DEEPSEEK_API_KEY=\"sk-...\" && python3 test.py"
    )
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Hello! Who are you? Introduce yourself in detail."}
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)