# 本文件需要 OPENAI_API_KEY，以及本机准备好 Gmail 的 credentials.json（客户端秘密文件）与
# 授权的 token.json；运行时会从已配置的 Gmail 账号读取邮件。需要开放 Google Cloud Gmail API。
import os, sys
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('DEEPSEEK_API_KEY')

# LangChain 1.x：Gmail 工具包移动到 langchain_community.agent_toolkits
from langchain_community.agent_toolkits import GmailToolkit

# 初始化Gmail工具包
toolkit = GmailToolkit()

# LangChain 1.x：gmail 工具从 langchain_community.tools.gmail.utils 导入
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

# 获取Gmail API的凭证，并指定相关的权限范围
credentials = get_gmail_credentials(
    token_file="token.json",  # Token文件路径
    scopes=["https://mail.google.com/"],  # 具有完全的邮件访问权限
    client_secrets_file="credentials.json",  # 客户端的秘密文件路径
)
# 使用凭证构建API资源服务
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# 获取工具
tools = toolkit.get_tools()
print(tools)

# LangChain 1.x：导入聊天模型与 create_agent，initialize_agent/AgentType 已删除
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 初始化聊天模型
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 通过指定的工具和聊天模型初始化 agent
agent = create_agent(llm, tools=toolkit.get_tools())

# 使用agent运行一些查询或指令（create_agent 输入为 {"messages": [HumanMessage(...)]}）
result = agent.invoke({
    "messages": [HumanMessage(content="总结一下最近的十条邮件有哪些需要我注意的事情")],
})

# 打印结果（聊天模型返回 AIMessage，取 .content）
print(result["messages"][-1].content)
