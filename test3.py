
import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 读取 API Key(兼容 DEEPSEEK_API_KEY 和 OPENAI_API_KEY 两种写法)
api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("Missing API key. Please set DEEPSEEK_API_KEY in .env and retry.")
    exit(1)

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# 加载Documents
base_dir = './OneFlower' # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
# 注意:LangChain 1.x 中 Text Splitter 已独立为 langchain_text_splitters 包
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
# 注意:LangChain 1.x 中 Qdrant 集成已独立为 langchain_qdrant 包
from langchain_qdrant import Qdrant
# DeepSeek 目前只提供对话模型(deepseek-v4-flash / deepseek-v4-pro)、没有Embedding API,
# 所以嵌入用本地模型 BAAI/bge-small-zh-v1.5(首次运行自动下载约95MB,之后离线可用)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=embeddings, # 用DeepSeek的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain_openai import ChatOpenAI # ChatOpenAI模型
# LangChain 1.x 中旧版链与检索器已迁移到 langchain_classic 包
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain_classic.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain_classic.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个大模型工具 - DeepSeek 的 deepseek-v4-flash
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)

# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__, template_folder='template') # Flask APP(注意:模板目录是单数的 template)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')

        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain.invoke({"query": question})

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == "__main__":
    # 注意:macOS 的 AirPlay 接收器占用 5000 端口,所以这里用 5001
    app.run(host='0.0.0.0',debug=True,port=5001)
