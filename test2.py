import os
os.environ["OPENAI_API_KEY"] = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY')
from langchain.chat_models import init_chat_model

chat= init_chat_model(
    "deepseek-v4-flash",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,  # Default; increase for unreliable networks
)
from langchain.messages import HumanMessage, AIMessage, SystemMessage
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]
response = chat(messages)
print(response)