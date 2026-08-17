from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
api_key = os.environ.get('DEEPSEEK_API_KEY')

# LangChain 1.x：用 RunnableLambda + llm 实现"摘要 + 窗口缓冲"记忆。
# 相当于旧 ConversationSummaryBufferMemory：历史过长时，
#   - 较早的部分压缩成一段摘要（摘要记忆），
#   - 最近 k 轮保留原文（窗口记忆）。
# 两者合起来塞进 history，既省 token 又不至于丢掉关键信息。
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

# 初始化大语言模型
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0.5,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个花店店员，请根据对话历史来回答。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

_store = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

with_history = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input", history_messages_key="history",
)

# 摘要链
summary_prompt = PromptTemplate.from_template(
    "把下面这段多轮对话压缩成一段简洁的中文摘要，只保留关键信息：\n{text}\n摘要:"
)
summary_chain = summary_prompt | llm | StrOutputParser()

# 超过 MAX_MESSAGES 条消息就触发摘要；最近 K 轮（K_ROUNDS）保留原文
MAX_MESSAGES = 6
K_ROUNDS = 1  # 窗口：最近保留 1 轮原文（2 条消息）

def summarize_and_answer(input_text: str, session_id: str) -> str:
    """历史过长时：早期压成摘要，最近 K 轮保留原文，然后交给模型回答。"""
    hist = get_history(session_id)
    msgs = hist.messages

    if len(msgs) > MAX_MESSAGES:
        # 需要被"挖掉"的较早部分：除最近 K_ROUNDS 轮之外的所有消息
        old = msgs[:-K_ROUNDS * 2]
        recent = msgs[-K_ROUNDS * 2:]

        transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
        summary = summary_chain.invoke({"text": transcript})
        print(f"[摘要+缓冲] 历史过长({len(msgs)}条)，早期{len(old)}条压成摘要，保留最近{K_ROUNDS}轮原文...")

        # 用 摘要(System) + 最近窗口原文 重建历史
        new_hist = InMemoryChatMessageHistory()
        new_hist.add_message(SystemMessage(content="早前对话的摘要：\n" + summary))
        for m in recent:
            new_hist.add_message(m)
        _store[session_id] = new_hist

    return with_history.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}},
    )

answer_chain = RunnableLambda(lambda d: summarize_and_answer(d["input"], d["session_id"]))

session = {"session_id": "s1"}
print("回合1：")
print(answer_chain.invoke({"input": "我姐姐明天要过生日，我需要一束生日花束。", **session}))
print("回合2：")
print(answer_chain.invoke({"input": "她喜欢粉色玫瑰，颜色是粉色的。", **session}))
print("回合3：")
print(answer_chain.invoke({"input": "我喜欢蓝色绣球，记住了吗？", **session}))
print("回合4：")
print(answer_chain.invoke({"input": "下周我要给朋友买粉玫瑰。", **session}))
# 历史超过 MAX_MESSAGES，触发"摘要 + 保留最近窗口"
print("回合5：")
print(answer_chain.invoke({"input": "你还记得我姐姐喜欢什么颜色的花吗？", **session}))

print("\n最终 store 中的历史（摘要 + 最近窗口原文）：")
for m in _store[session["session_id"]].messages:
    print(f"  [{m.type}] {m.content}")