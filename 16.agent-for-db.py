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

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)


@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query against the flower shop database and return the rows."""
    return str(db.run(sql))


# 创建SQL智能体（create_sql_agent/SQLDatabaseToolkit 已删除，改用 create_agent + SQL工具）
agent = create_agent(llm, tools=[query_database], debug=True)

# 使用Agent执行SQL查询
questions = [
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
]

for question in questions:
    out = agent.invoke({"messages": [HumanMessage(content=question)]})
    response = out["messages"][-1].content
    print(response)