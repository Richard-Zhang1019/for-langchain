import os, sys
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.environ.get('SERP_API_KEY')

# 规避 langgraph 执行过程中关闭 sys.stdout 的 bug：提前备份 stdout 的文件描述符
_saved_stdout_fd = os.dup(1)

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SerpAPIWrapper

# 初始化大模型（LangChain 1.x 用 ChatOpenAI 聊天模型，默认现役 gpt-4o-mini）
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 封装 SerpAPI 搜索工具（必须带 docstring 和类型注解）
_search = SerpAPIWrapper()

@tool
def search_web(query: str) -> str:
    """Search the web (SerpAPI) for current information. Use for any question requiring fresh facts, such as market prices."""
    return _search.run(query)

# 封装数学计算工具（eval 仅计算数值表达式，禁用内建函数）
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric/math expression like '25 * 1.15'. Returns the result."""
    return str(eval(expression, {"__builtins__": {}}, {}))

# 初始化智能体：模型用 tool-calling 自主决定使用搜索或计算工具
agent = create_agent(llm, tools=[search_web, calculator])

# 跑起来（create_agent 的输入是 {"messages": [HumanMessage(...)]}，不是 {"question": ...}）
out = agent.invoke({
    "messages": [HumanMessage(content="目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")],
})
# 恢复被 langgraph 关闭的 sys.stdout，再输出最终答案
sys.stdout = os.fdopen(_saved_stdout_fd, "w", buffering=1)
print(out["messages"][-1].content)