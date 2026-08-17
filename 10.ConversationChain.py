from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
api_key = os.environ.get('DEEPSEEK_API_KEY')

# LangChain 1.x：旧 ConversationChain 已移除，改用 RunnableWithMessageHistory
# 结合 MessagesPlaceholder 实现多轮对话记忆。
# 思路：
#   1) 提示模板里放一个 MessagesPlaceholder("history") 占位历史消息；
#   2) 用 InMemoryChatMessageHistory 存每个会话(session_id)的历史；
#   3) RunnableWithMessageHistory 在每次调用时把 history 填进模板、并把对话消息写回存储。
import sys
sys.path.insert(0, "..")  # 让相对位置可读

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 初始化大语言模型
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0.5,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 对话提示模板：system 承担总能；history 是历史消息；{input} 是用户本轮输入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手，请根据对话历史来作答。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 基础链：提示模板 -> 模型 -> 字符串解析
chain = prompt | llm | StrOutputParser()

# 会话历史存储：键为 session_id
_store = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

# 包上带记忆的链
with_history = RunnableWithMessageHistory(
    chain, 
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 打印对话的提示模板结构（相当于旧 ConversationChain 打印 prompt.template）
print("对话提示模板结构：")
for msg in prompt.messages:
    print("  -", msg)

print("\n演示：")
# 第一轮
r1 = with_history.invoke(
    {"input": "你好，我叫小明。"},
    config={"configurable": {"session_id": "s1"}},
)
print("第一轮回答:", r1)

# 第二轮：模型能通过 history 记住上一轮内容
r2 = with_history.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "s1"}},
)
print("第二轮回答:", r2)

print("\n历史存储内容：")
for m in _store["s1"].messages:
    print(f"  [{m.type}] {m.content}")