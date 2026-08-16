from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
api_key = os.environ.get('DEEPSEEK_API_KEY')

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 初始化大模型
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0.7,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 第一个子链：根据花名和颜色生成鲜花的介绍（输出到键 introduction）
template = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:"""
prompt_template = PromptTemplate(
    input_variables=["name", "color"],
    template=template
)
introduction_chain = prompt_template | llm | StrOutputParser()

# 第二个子链：根据鲜花介绍写出评论（输入 introduction，输出到键 review）
template = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
鲜花介绍:
{introduction}
花评人对上述花的评论:"""
prompt_template = PromptTemplate(
    input_variables=["introduction"],
    template=template
)
review_chain = prompt_template | llm | StrOutputParser()

# 第三个子链：根据介绍和评论写自媒体文案（输入 introduction、review，输出 social_post_text）
template = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}
社交媒体帖子:
"""
prompt_template = PromptTemplate(
    input_variables=["introduction", "review"],
    template=template
)
social_post_chain = prompt_template | llm | StrOutputParser()

# 串联成链：RunnablePassthrough.assign 会在每步把上一步的结果字典"追加"进当前上下文，
# 因此后面子链能以键名直接取到前面链的输出（等价于旧 SequentialChain 的逐链传参）
chain = (
    {"introduction": introduction_chain}   # 输入 {"name","color"}，经子链得到 introduction
    | RunnablePassthrough.assign(review=lambda d: review_chain.invoke({"introduction": d["introduction"]}))
    | RunnablePassthrough.assign(social_post_text=lambda d: social_post_chain.invoke(
        {"introduction": d["introduction"], "review": d["review"]}))
)

# 运行链并打印结果
result = chain.invoke({
    "name": "玫瑰",
    "color": "黑色"
})

# 结果字典里三个键都保留了下来，逐个打印
print("鲜花介绍:\n", result["introduction"])
print("\n花评:\n", result["review"])
print("\n社交媒体文案:\n", result["social_post_text"])