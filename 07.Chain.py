# # ------Part 1
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
api_key = os.environ.get('DEEPSEEK_API_KEY')

# #----第一步 创建提示
# # 导入LangChain中的提示模板
# from langchain_core.prompts import PromptTemplate
# # 原始字符串模板
# template = "{flower}的花语是?"
# # 创建LangChain模板
# prompt_temp = PromptTemplate.from_template(template) 
# # 根据模板创建提示
# prompt = prompt_temp.format(flower='玫瑰')
# # 打印提示的内容
# print(prompt)

# #----第二步 创建并调用模型 
# # 导入LangChain中的OpenAI模型接口
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(
#     model_name="deepseek-v4-flash",
#     temperature=0,
#     base_url="https://api.deepseek.com",
#     api_key=api_key,
# )

# # 传入提示，调用模型，返回结果
# result = model.invoke(prompt)
# print(result)

# Part2 使用链调用（新版 LangChain 用 LCEL 表达式构建链）
# 导入所需的库
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# 原始字符串模板
template = "{flower}在{season}的花语是?"
prompt = PromptTemplate.from_template(template)

# 创建模型实例
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 用 | 运算符把提示模板和模型组合成链
chain = prompt | llm
# 调用链，返回结果
result = chain.invoke({"flower": "玫瑰", "season": "春季"})
print(result)