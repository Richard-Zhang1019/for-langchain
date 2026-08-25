# 需要 OPENAI_API_KEY（可通过环境变量或 .env 提供）
# 运行前请先执行 16_操作数据库/01_DBCreation.py 创建 FlowerShop.db
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 连接到FlowerShop数据库（sqlite:/// 为相对当前工作目录的路径）
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")


@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query against the flower shop database and return the rows."""
    return str(db.run(sql))


# 创建一个ChatOpenAI实例，这里我们设置温度为0，意味着模型输出会更加确定性
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 语义仍为"LLM 生成 SQL 再执行"：create_agent + query_database 工具
agent = create_agent(llm, tools=[query_database])

# 运行与鲜花运营相关的问题
questions = [
    "有多少种不同的鲜花？",
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
    "从法国进口的鲜花有多少种？",
    "哪种鲜花的销售量最高？",
]

for q in questions:
    out = agent.invoke({"messages": [HumanMessage(content=q)]})
    answer = out["messages"][-1].content
    print(f"问题：{q}\n答案：{answer}\n")