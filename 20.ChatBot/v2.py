# 设置OpenAI API密钥
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')

# 导入所需的库和模块
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# 会话历史存储（session_id -> 历史）
_history_store = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """按 session_id 返回（或新建）一段对话历史"""
    if session_id not in _history_store:
        _history_store[session_id] = InMemoryChatMessageHistory()
    return _history_store[session_id]


def format_docs(docs):
    """把检索到的文档列表拼接成上下文文本"""
    return "\n\n".join(d.page_content for d in docs)


# ChatBot类的实现-带检索功能
class ChatbotWithRetrieval:

    def __init__(self, dir):

        # 加载Documents
        base_dir = dir # 文档的存放目录
        documents = []
        for file in os.listdir(base_dir): 
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)
        
        # 向量数据库（Qdrant 内存模式，无需 qdrant 服务）
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits, # 以分块的文档
            embedding=OpenAIEmbeddings(), # 用OpenAI的Embedding Model做嵌入
            path=":memory:",  # in-memory 存储
            collection_name="my_documents",) # 指定collection_name
        
        # 初始化LLM
        self.llm = ChatOpenAI(
          model_name="deepseek-v4-flash",
          base_url="https://api.deepseek.com",
          api_key=api_key,
        )

        # 设置Retrieval Chain
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 构建"RAG + 历史"问答链（替代 ConversationalRetrievalChain + ConversationSummaryMemory）
        prompt = ChatPromptTemplate.from_messages([
            ("system", "使用下面的上下文来回答用户关于鲜花的提问；如果上下文中没有答案，请如实说明。\n\n上下文：\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        def _retrieve_with_question(inputs):
            """根据用户的问题检索上下文文本"""
            return format_docs(self.retriever.invoke(inputs["question"]))

        chain = (
            {"context": _retrieve_with_question, "question": lambda x: x["question"]}
            | prompt | self.llm | StrOutputParser()
        )
        self.qa = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_answer(self, user_input, session_id: str = "s1"):
        """针对用户的输入，结合检索结果和历史给出回答"""
        return self.qa.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )

    # 交互对话的函数
    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
            # 调用 Retrieval Chain  
            answer = self.get_answer(user_input)
            print(f"Chatbot: {answer}")

if __name__ == "__main__":
    # 启动Chatbot（需要 OneFlower 数据目录）
    folder = "OneFlower"
    bot = ChatbotWithRetrieval(folder)
    bot.chat_loop()