# agent/sql_agent.py
import ast
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "Chinook.db"


class SQLCaptureHandler(BaseCallbackHandler):
    """在 Agent 调用 sql_db_query 工具时，截获它执行的 SQL"""

    def __init__(self):
        self.sql = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "")
        if tool_name != "sql_db_query":
            return

        # 情况 1：直接就是 dict
        if isinstance(input_str, dict):
            self.sql = input_str.get("query")
            return

        s = str(input_str)

        # 情况 2：JSON 字符串（双引号）
        try:
            data = json.loads(s)
            if isinstance(data, dict) and "query" in data:
                self.sql = data["query"]
                return
        except (json.JSONDecodeError, TypeError):
            pass

        # 情况 3：Python dict 的字符串表示（单引号）
        try:
            data = ast.literal_eval(s)
            if isinstance(data, dict) and "query" in data:
                self.sql = data["query"]
                return
        except (ValueError, SyntaxError):
            pass

        # 兜底：当作纯 SQL 字符串
        self.sql = s


def create_llm():
    """创建 LLM 实例，供 Agent 和报告生成共用"""
    return ChatOpenAI(
        model="qwen3.7-flash",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


def create_agent():
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    llm = create_llm()

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent, db


def run_agent(agent, message):
    """调用 Agent，同时截获它执行的 SQL，返回 (文字回答, SQL)"""
    handler = SQLCaptureHandler()
    result = agent.invoke({"input": message}, config={"callbacks": [handler]})
    return result["output"], handler.sql


def generate_report(llm, question, sql, df_preview):
    """调用 LLM，根据问题、SQL 和查询结果写一份 Markdown 分析报告"""
    prompt = f"""你是一位数据分析师。请根据下面的信息，写一份简洁的 Markdown 分析报告。

用户问题：{question}

执行的 SQL：
{sql}

查询结果（前 20 行）：
{df_preview}

要求：
1. 使用 Markdown 格式，包含三个部分：数据概览、关键发现、结论
2. 语言简洁专业，重点突出
3. 控制在 300 字以内
4. 直接输出 Markdown 正文，不要写“好的”之类的开场白，不要用一级标题
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


if __name__ == "__main__":
    agent, db = create_agent()
    answer, sql = run_agent(agent, "数据库里一共有多少首歌曲？")
    print("\n最终回答：", answer)
    print("捕获到的 SQL：", sql)