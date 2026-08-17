import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
api_key = os.environ.get('DEEPSEEK_API_KEY')

# =========================================================
# 路由（Router）思想：先用一个模型对输入做"分类"，判断它属于哪个场景，
# 再把输入派发到对应场景的目标链去回答。
# 这是 LangChain 1.x 对旧 MultiPromptChain / LLMRouterChain 的替代写法。
# =========================================================
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI

# 目标链统一使用的聊天模型
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 两个场景的提示模板
flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

# 分类器模型：temperature=0 尽量稳定，只让它返回三类之一
router_llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)
router_prompt = PromptTemplate.from_template(
    "请判断下面这个问题属于哪个类别，只返回 flower_care、flower_decoration 或 default 中的一个词。\n"
    "问题: {input}\n类别:"
)
router_chain = router_prompt | router_llm | StrOutputParser()

# 构建目标链（两个场景 + 一个默认兜底链）
flower_care_chain = PromptTemplate.from_template(flower_care_template) | llm | StrOutputParser()
flower_deco_chain = PromptTemplate.from_template(flower_deco_template) | llm | StrOutputParser()

default_template = "请直接回答这个问题：\n{input}"
default_chain = PromptTemplate.from_template(default_template) | llm | StrOutputParser()


# 方式一（推荐，直观）：一个函数做分类并派发，用 RunnableLambda 包起来
def route(text: str) -> str:
    """先让分类模型给出类别，再把原问题交给对应目标链。"""
    category = router_chain.invoke({"input": text}).strip().lower()
    if "flower_care" in category:
        return flower_care_chain.invoke({"input": text})
    elif "flower_decoration" in category:
        return flower_deco_chain.invoke({"input": text})
    else:
        return default_chain.invoke({"input": text})


chain = RunnableLambda(route)


# 方式二（效果相同，供参考）：用 RunnableBranch 把"条件 + 分支链"配对，第三个参数是默认分支。
# 每个条件既是谓词函数（返回布尔），又是分支（这里是把新输入重新 invoke 目标链）
def _branch_care(x: str) -> str:
    return flower_care_chain.invoke({"input": x})


def _branch_deco(x: str) -> str:
    return flower_deco_chain.invoke({"input": x})


def _branch_default(x: str) -> str:
    return default_chain.invoke({"input": x})


# chain_branch = RunnableBranch(
#     (lambda x: router_chain.invoke({"input": x}).strip().lower() == "flower_care", _branch_care),
#     (lambda x: router_chain.invoke({"input": x}).strip().lower() == "flower_decoration", _branch_deco),
#     _branch_default,
# )

print("=" * 60)
# 测试1：应进入 flower_care（养花护理）
print("问题: 如何为玫瑰浇水？")
print(chain.invoke("如何为玫瑰浇水？"))
print("=" * 60)
# 测试2：应进入 flower_decoration（鲜花装饰）
print("问题: 如何为婚礼场地装饰花朵？")
print(chain.invoke("如何为婚礼场地装饰花朵？"))
print("=" * 60)
# 测试3：不属于任何场景，应进入 default 兜底链
print("问题: 如何区分阿豆和罗豆？")
print(chain.invoke("如何区分阿豆和罗豆？"))