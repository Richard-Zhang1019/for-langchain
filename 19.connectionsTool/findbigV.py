# 从上级目录加载 .env 中的 API 密钥
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# .env 中使用 SERP_API_KEY，而 SerpAPIWrapper 读取 SERPAPI_API_KEY，做一次映射
os.environ.setdefault("SERPAPI_API_KEY", os.environ.get("SERP_API_KEY", ""))

# 导入所取的库
import re
from agents.weibo_agent import lookup_V
from tools.general_tool import remove_non_chinese_fields
from tools.scraping_tool import get_data
from tools.textgen_tool import generate_letter


def find_bigV(flower: str) :
    # 拿到UID
    response_UID = lookup_V(flower_type = flower )

    # 抽取UID里面的数字
    UID = re.findall(r'\d+', response_UID)[0]
    print("这位鲜花大V的微博ID是", UID)

    # 根据UID爬取大V信息
    person_info = get_data(UID)
    print(person_info)

    # 移除无用的信息
    remove_non_chinese_fields(person_info)
    print(person_info)

    # 调用函数根据大V信息生成文本
    result = generate_letter(information = person_info)
    print(result)

    return result


if __name__ == "__main__":

    # 拿到UID
    response_UID = lookup_V(flower_type = "牡丹" )

    # 抽取UID里面的数字
    UID = re.findall(r'\d+', response_UID)[0]
    print("这位鲜花大V的微博ID是", UID)

    # 根据UID爬取大V信息
    person_info = get_data(UID)
    print(person_info)

    # 移除无用的信息
    remove_non_chinese_fields(person_info)
    print(person_info)

    result = generate_letter(information = person_info)
    print(result)

    import json
    # 使用json.loads将字符串解析为字典
    result = json.loads(result)
    # 注意：jsonify 必须在 Flask 应用上下文中使用，命令行直接测试时打印字典即可
    abc = {
        "summary": result["summary"],
        "facts": result["facts"],
        "interest": result["interest"],
        "letter": result["letter"],
    }
    print(abc)


