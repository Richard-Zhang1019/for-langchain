# 导入一个搜索UID的工具
from tools.search_tool import get_UID

# 需要 OPENAI_API_KEY（OpenAI 模型调用）与 SERPAPI_API_KEY（SerpAPI 搜索获取微博 UID）
# 导入所需的库
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

# 通过LangChain代理找到UID的函数
def lookup_V(flower_type: str) -> str:
    # 初始化大模型
    llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0, base_url="https://api.deepseek.com",)    

    # 寻找UID的模板
    template = """given the {flower} I want you to get a related 微博 UID.
                  Your answer should contain only a UID.
                  The URL always starts with https://weibo.com/u/
                  for example, if https://weibo.com/u/1669879400 is her 微博, then 1669879400 is her UID
                  This is only the example don't give me this, but the actual UID"""
    # 完整的提示模板
    prompt_template = PromptTemplate(
        input_variables=["flower"], template=template
    )

    # 代理的工具（用 langchain_core.tools.Tool 对象，带 name/description 即可）
    # 注意：工具名只能包含字母、数字、下划线和连字符，不能含空格或中文
    my_tool = Tool(
        name="crawl_google_for_weibo_uid",
        func=get_UID,
        description="useful for when you need get the 微博 UID",
    )

    # 初始化代理
    agent = create_agent(llm, tools=[my_tool], debug=True)

    # 返回找到的UID（create_agent 输入为 {"messages": [HumanMessage(...)]}）
    result = agent.invoke({"messages": [HumanMessage(content=prompt_template.format_prompt(flower=flower_type).to_string())]})

    return result["messages"][-1].content