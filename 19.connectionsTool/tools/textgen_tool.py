# 需要 OPENAI_API_KEY（OpenAI 模型生成文案）
# 导入所需要的库
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from tools.parsing_tool import letter_parser


def _strip_code_fences(text: str) -> str:
    """去掉模型输出中包裹的 ```json ... ``` 等 markdown 代码块围栏，返回纯 JSON 文本。"""
    # 匹配开头可选的 ```json / ``` 和结尾的 ```
    text = re.sub(r"^\s*```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    return text.strip()


# 生成文案的函数
def generate_letter(information):

    # 设计提示模板
    letter_template = """
         下面是这个人的微博信息 {information}
         请你帮我:
         1. 写一个简单的总结
         2. 挑两件有趣的特点说一说
         3. 找一些他比较感兴趣的事情
         4. 写一篇热情洋溢的介绍信
         \n{format_instructions}"""
    
    prompt_template = PromptTemplate(
        input_variables=["information"],
        template=letter_template,
        partial_variables={
            "format_instructions": letter_parser.get_format_instructions()
        },         
    )

    # 初始化大模型
    llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0, base_url="https://api.deepseek.com",)    

    # 初始化链（LCEL 取代 LLMChain）
    chain = prompt_template | llm | StrOutputParser()

    # 生成文案（返回纯 JSON 字符串，保持与下游 json.loads 兼容）
    result = chain.invoke({"information": information})
    return _strip_code_fences(result)