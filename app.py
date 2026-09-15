# app.py
import base64
import os
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

from agent.sql_agent import (
    create_agent,
    create_llm,
    run_agent,
    generate_report,
    DB_PATH,
)

# ============ 初始化 ============
print("正在初始化 Agent...")
agent, db = create_agent()
llm = create_llm()
engine = create_engine(f"sqlite:///{DB_PATH}")
print("Agent 就绪")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 保存最近一次查询状态，供“生成报告”按钮使用
last_query = {"question": None, "sql": None, "df": None, "fig": None}


# ============ 核心逻辑 ============
def try_make_chart(sql):
    """执行 SQL，返回 (DataFrame, Plotly图或None)"""
    if not sql:
        return None, None
    try:
        df = pd.read_sql_query(sql, engine)
    except Exception as e:
        print(f"[图表] SQL 执行失败：{e}")
        return None, None

    if len(df) < 2 or len(df.columns) < 2:
        return df, None

    x_col, y_col = df.columns[0], df.columns[1]
    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} / {x_col}")
    return df, fig


def respond(message, chat_history):
    """用户发送消息时调用"""
    if not message.strip():
        return "", chat_history, None, "", None

    answer, sql = run_agent(agent, message)
    df, fig = try_make_chart(sql)

    last_query.update({"question": message, "sql": sql, "df": df, "fig": fig})

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", chat_history, fig, "", None


def build_report():
    """点击“生成报告”按钮时调用"""
    if not last_query["question"]:
        return "请先提问一个问题，再生成报告。", None

    question = last_query["question"]
    sql = last_query["sql"] or "（未捕获到 SQL）"
    df = last_query["df"]
    fig = last_query["fig"]

    df_preview = df.head(20).to_string(index=False) if df is not None else "（无数据）"
    analysis = generate_report(llm, question, sql, df_preview)

    report = f"# 数据分析报告\n\n**问题**：{question}\n\n{analysis}\n"

    # 嵌入图表 PNG
    if fig is not None:
        try:
            img_bytes = fig.to_image(format="png", width=900, height=500)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            report += f"\n## 可视化图表\n\n![图表](data:image/png;base64,{img_b64})\n"
        except Exception as e:
            print(f"[报告] 图表转 PNG 失败：{e}")
            report += f"\n## 可视化图表\n\n（图表转换失败：{e}）\n"

    # 保存为 .md 文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = str(REPORTS_DIR / f"report_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[报告] 已保存到 {filepath}")
    return report, filepath


# ============ 界面 ============
with gr.Blocks(title="数据分析 Agent") as demo:
    gr.Markdown("# 数据分析 Agent")
    gr.Markdown("用自然语言查询 Chinook 数据库，支持自动生成图表和分析报告。")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420, label="对话")
            msg = gr.Textbox(
                placeholder="输入问题，按回车发送...",
                label="你的问题",
                lines=1,
            )
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空")

            gr.Markdown("---")
            report_btn = gr.Button("生成报告", variant="secondary")
            report_md = gr.Markdown(label="分析报告")
            report_file = gr.File(label="下载报告 (.md)")

        with gr.Column(scale=2):
            plot = gr.Plot(label="自动生成的图表")

    msg.submit(respond, [msg, chatbot], [msg, chatbot, plot, report_md, report_file])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot, plot, report_md, report_file])
    clear_btn.click(
        lambda: (None, None, "", None),
        None,
        [chatbot, plot, report_md, report_file],
    )
    report_btn.click(build_report, None, [report_md, report_file])


if __name__ == "__main__":
    demo.launch()