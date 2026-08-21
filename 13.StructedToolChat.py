import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.environ.get('SERP_API_KEY')

# LangChain 1.x：游戏浏览器工具改到 langchain_community；同步版浏览器
# create_async_playwright_browser → create_sync_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser

browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
tools = toolkit.get_tools()
print(tools)

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# LLM不稳定，对于这个任务，可能要多跑几次才能得到正确结果
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0.5,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

agent = create_agent(llm, tools=tools)

# 同步运行（LangChain 1.x 用 agent.invoke 取最终答案；输入为 {"messages": [HumanMessage(...)]}）
response = agent.invoke({"messages": [HumanMessage(content="What are the headers on python.langchain.com?")]})
print(response["messages"][-1].content)