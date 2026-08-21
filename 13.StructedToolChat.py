import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.environ.get('SERP_API_KEY')

# LangChain 1.x：Playwright 同步浏览器基于 greenlet，绑定创建线程，
# 而 create_agent 底层的 ToolNode 会在线程池里并发执行工具，
# 会触发 greenlet.error: Cannot switch to a different thread。
# 解决：改用异步浏览器 + agent.ainvoke，全程在事件循环单线程中运行。
#
# 注意：langchain_community 提供的 create_async_playwright_browser() 内部用
# run_until_complete，在已有事件循环里调用会报 "This event loop is already running"，
# 因此这里直接用 playwright.async_api.async_playwright() 在事件循环内启动浏览器。
from playwright.async_api import async_playwright

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


async def main():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    tools = toolkit.get_tools()
    print(tools)

    # LLM不稳定，对于这个任务，可能要多跑几次才能得到正确结果
    llm = ChatOpenAI(
        model_name="deepseek-v4-flash",
        temperature=0.5,
        base_url="https://api.deepseek.com",
        api_key=api_key,
    )

    agent = create_agent(llm, tools=tools)

    # LangChain 1.x：异步执行，工具走 _arun，规避线程切换导致的 greenlet 报错
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="What are the headers on python.langchain.com?")]}
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
