# 本文件需要 OPENAI_API_KEY 才能运行
# 设置OpenAI API密钥
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')

import asyncio
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain 1.x：不再使用 ConversationChain / ConversationBufferMemory。
# 改用 ChatOpenAI + RunnableWithMessageHistory 实现带记忆的对话（指南第6节）。
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0.5,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一家鲜花店的客服，语气亲切。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

# 会话记忆存储
_store = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

with_history = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input", history_messages_key="history",
)

# 使用context manager进行token counting
with get_openai_callback() as cb:
    # 第一天的对话
    # 回合1
    with_history.invoke(
        {"input": "我姐姐明天要过生日，我需要一束生日花束。"},
        config={"configurable": {"session_id": "s1"}},
    )
    print("第一次对话后的历史:", _store["s1"].messages)

    # 回合2
    with_history.invoke(
        {"input": "她喜欢粉色玫瑰，颜色是粉色的。"},
        config={"configurable": {"session_id": "s1"}},
    )
    print("第二次对话后的历史:", _store["s1"].messages)

    # 回合3 （第二天的对话，模型应记得之前的购买需求）
    reply = with_history.invoke(
        {"input": "我又来了，还记得我昨天为什么要来买花吗？"},
        config={"configurable": {"session_id": "s1"}},
    )
    print("\n第三次对话后的回复:", reply)

# 输出使用的tokens
print("\n总计使用的tokens:", cb.total_tokens)

# 进行更多的异步交互和token计数
async def additional_interactions():
    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.ainvoke("我姐姐喜欢什么颜色的花？") for _ in range(3)]
        )
    print("\n另外的交互中使用的tokens:", cb.total_tokens)

# 运行异步函数
asyncio.run(additional_interactions())