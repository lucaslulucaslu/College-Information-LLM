"""Forward Pathway AI Chatbot."""

import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, PercentFormatter

from utilities import languages
from utilities.colleges import CollegesData
from utilities.knowledgebase import TXTKnowledgeBase
from utilities.ranking import CollegeRanking
from utilities.schema import (College_Info, CollegeRouter, GraphState,
                              RankingType, RouteQuery)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="error.log", encoding="utf-8", level=logging.ERROR)

st.set_page_config(
    page_title="Forward Pathway AI Chatbot",
    page_icon="./logos/fp_logo.png",
    menu_items={
        "About": "APP资料及数据来源为美国续航教育官网，输出内容经ChatGPT整理，APP测试阶段回答不一定准确，请确认后使用"
    },
)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chat.forwardpathway.com"

SEARCH_DOCS_NUM = 4
SEARCH_COLLEGES_NUM = 2

# Choose LLM Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# choose language
lang_index = "lang" in st.query_params and st.query_params["lang"].upper() == "EN"

language_question = ["语言选择：", "Languages:"]


def change_language_fuc():
    """Delete all messages after change the language."""
    if "messages" in st.session_state:
        del st.session_state["messages"]


language = st.radio(
    label=language_question[lang_index],
    options=["CN", "EN"],
    horizontal=True,
    index=lang_index,
    on_change=change_language_fuc,
)
if language == "EN":
    lang_dict = languages.en_dict
else:
    lang_dict = languages.cn_dict
    font_path = ".//utilities//MicrosoftYaHei.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop.get_name()


def router_college(state: GraphState):
    """Route the user's question to the college database or others."""
    structured_llm = llm.with_structured_output(CollegeRouter, method="json_schema")
    system_message = """你是一名熟悉美国大学的专家，下面将给出用户的一个问题，你需要判断用户的问题是否为某所特定大学的相关问题，Yes为相关问题，\
        No为不与美国大学相关。比如：哈佛大学排名，普林斯顿录取率等这类只包含一个学校名称的问题则回答Yes，\
            而美国大学申请、美国留学难不难、会计专业排名、物理学排名等不包含美国大学名称的问题则回答No。"""
    llm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "用户问题如下：{question}"),
        ]
    )
    question_router = llm_prompt | structured_llm
    response = question_router.invoke({"question": state["question"]})
    return {"router_college_flag": response.college}


def router_college_func(state: GraphState):
    """Route the user's question to the college database or others."""
    if state["router_college_flag"] == "Yes":
        return "to_database"
    return "to_router_ranking"


def router_ranking(state: GraphState):
    """Route the user's question to the ranking database or others."""
    structured_llm_router = llm.with_structured_output(RouteQuery, method="json_schema")

    # Prompt
    system = """
        你是一位路径选择的专家，负责根据用户的问题及历史聊天记录，从以下两条路径中选择最合适的一条："vectorstore"、"ranking"。

    1. "vectorstore"：包含美国留学的综合资料，例如美国留学申请流程、转学事项等。如果用户的问题涉及美国留学的整体信息或涉及两所及以上大学的信息，请选择此路径。

    2. "ranking"：提供美国大学的排名数据，包括综合排名、学院排名、专业排名等。如果用户的问题是了解一组大学的排名列表，请选择此路径。比如美国大学排名，商科排名等等。

    请根据用户的问题内容和规则做出最合适的路径选择。
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "用户问题如下：{question}\n\n历史聊天记录如下：{chat_history}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    response = question_router.invoke(
        {"question": state["question"], "chat_history": state["chat_history"]}
    )
    return {"router_ranking_flag": response.router}


def router_ranking_func(state: GraphState):
    """Route the user's question to the ranking database or others."""
    if state["router_ranking_flag"] == "vectorstore":
        return "to_retrieve"
    elif state["router_ranking_flag"] == "ranking":
        return "to_ranking"


def retrieve(state: GraphState):
    """Retrieve the documents from the knowledge base."""
    # print("---RETRIEVE---")
    vector = TXTKnowledgeBase(
        txt_source_folder_path="lxbd"
    ).return_retriever_from_persistant_vector_db()
    vector_lxsq = TXTKnowledgeBase(
        txt_source_folder_path="lxsq"
    ).return_retriever_from_persistant_vector_db()
    vector_emergency = TXTKnowledgeBase(
        txt_source_folder_path="emergency"
    ).return_retriever_from_persistant_vector_db()
    vector.merge_from(vector_lxsq)
    vector.merge_from(vector_emergency)
    documents_retriever = vector.as_retriever(search_kwargs={"k": SEARCH_DOCS_NUM})
    question = state["question"]
    documents = documents_retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "chat_history": state["chat_history"],
    }


def generate(state: GraphState):
    """Generate the answer to the user's question."""
    # print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [("system", lang_dict["prompt_document"]), ("human", "{question}")]
    )

    rag_chain = prompt | llm | StrOutputParser()
    # RAG generation
    generation = rag_chain.stream(
        {
            "context": documents,
            "question": question,
            "chat_history": state["chat_history"],
        }
    )

    return {"documents": documents, "question": question, "generation": generation}


def get_college_info(state: GraphState):
    """Get the college information from the database."""
    # print("---COLLEGE NAME---")

    college_data = CollegesData()
    college_vector = college_data.return_colleges_vector_from_db()
    college_retriever = college_vector.as_retriever(
        search_kwargs={"k": SEARCH_COLLEGES_NUM}
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位了解美国高等院校的专家，你需要根据用户的问题及历史聊天记录提取出一所美国高等院校的全名，包括中文名和英文名，输出格式为'中文全名（英文全名）'",
            ),
            ("human", "用户问题如下：{question}\n\n历史聊天记录如下：{chat_history}"),
        ]
    )
    college_name_chain = prompt | llm | StrOutputParser()

    college_info_structured_output = llm.with_structured_output(
        College_Info, method="json_schema"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "基于下面学校信息内容及用户的问题和历史聊天记录，按照格式输出学校信息回答",
            ),
            (
                "human",
                "用户问题如下：{question}\n\n学校信息内容如下：{context}\n\n历史聊天记录如下：{chat_history}",
            ),
        ]
    )
    college_info_chain = prompt | college_info_structured_output

    question = state["question"]
    college_name = college_name_chain.invoke(
        {"question": question, "chat_history": state["chat_history"]}
    )
    college_info = college_info_chain.invoke(
        {
            "question": question,
            "context": college_retriever.invoke(college_name),
            "chat_history": state["chat_history"],
        }
    )
    return {"college_info": college_info, "question": question}


def database_router_func(state: GraphState):
    """Route the user's question to the college database or others."""
    post_id = state["college_info"].postid
    if (
        state["college_info"].data_type
        in {
            "排名",
            "录取率",
            "申请录取人数",
            "成绩要求",
            "学生组成",
            "学生人数",
            "学费",
            "毕业率",
            "犯罪率",
        }
        and len(post_id) > 0
    ):
        return "to_college_data_plot"
    else:
        return "to_retrieve"


def college_data_plot(state: GraphState):
    """Get the college data from the database."""
    question = state["question"]
    college_info = state["college_info"]
    dataURLs = {
        "rank_adm": "https://www.forwardpathway.com/d3v7/dataphp/school_database/ranking_admin_20240923.php?name=",
        "world_rank": "https://www.forwardpathway.com/d3v7/dataphp/chatbot/world_ranks4_20240605.php?name=",
        "score": "https://www.forwardpathway.com/d3v7/dataphp/school_database/score10_20231213.php?name=",
        "students": "https://www.forwardpathway.com/d3v7/dataphp/school_database/student_comp_20240118.php?name=",
        "students_number": "https://www.forwardpathway.com/d3v7/dataphp/school_database/international_students_20240118.php?name=",
        "info": "https://www.forwardpathway.com/d3v7/dataphp/school_database/school_information_20240821.php?name=",
        "crime": "https://www.forwardpathway.com/d3v7/dataphp/school_database/crime_yearly_20240324.php?name=",
    }
    if college_info.data_type == "排名":
        college_df = pd.read_json(dataURLs["rank_adm"] + str(college_info.postid))
        data = pd.read_json(dataURLs["world_rank"] + str(college_info.postid))
        # college_df['year']=college_df['year'].astype(str)
        college_df = college_df[["year", "rank"]].rename(
            columns={"year": "年", "rank": "USNews排名"}
        )
        college_df.loc[len(college_df)] = pd.Series(
            {"年": college_df["年"].to_list()[-1] + 1}
        ).astype("Int64")
        college_df.set_index("年", inplace=True)
        for index, row in data.iterrows():
            college_df[row["type"] + "世界大学排名"] = (
                pd.DataFrame(row["data"])
                .fillna(0)
                .rename(columns={"year": "年", "rank": (row["type"] + "世界大学排名")})
                .set_index("年")[(row["type"] + "世界大学排名")]
            )
            college_df[row["type"] + "世界大学排名"] = college_df[
                row["type"] + "世界大学排名"
            ].astype("Int64")
        college_df.dropna(how="all", inplace=True)
    elif college_info.data_type == "录取率":
        college_df = pd.read_json(dataURLs["rank_adm"] + str(college_info.postid))
        college_df["year"] = college_df["year"].astype(str)
        college_df = college_df[["year", "rate", "rate2"]].rename(
            columns={"year": "年", "rate": "男生录取率", "rate2": "女生录取率"}
        )
        college_df.set_index("年", inplace=True)
    elif college_info.data_type == "申请录取人数":
        college_df = pd.read_json(dataURLs["rank_adm"] + str(college_info.postid))
        college_df["year"] = college_df["year"].astype(str)
        college_df = college_df[["year", "enroll", "defer", "deny"]].rename(
            columns={
                "year": "年",
                "enroll": "录取且入学人数",
                "defer": "录取但未入学人数",
                "deny": "拒绝人数",
            }
        )
        college_df.set_index("年", inplace=True)
    elif college_info.data_type == "成绩要求":
        data = pd.read_json(dataURLs["score"] + str(college_info.postid))
        college_df = pd.DataFrame(data.columns, columns=["year"])
        testArray = {
            "SATR": "SAT阅读",
            "SATM": "SAT数学",
            "ACTC": "ACT综合",
            "ACTE": "ACT英语",
            "ACTM": "ACT数学",
        }
        for test in testArray.keys():
            college_df[test + "25"] = None
            college_df[test + "75"] = None
        college_df.set_index("year", inplace=True)

        def temp_fuc(x, item_name, item_per):
            for item in x:
                if item["name"] == item_name:
                    return item[item_per]

        for test, test_name in testArray.items():
            college_df[test + "25"] = data.T["score"].apply(
                temp_fuc, args=(test, "start")
            )
            college_df[test + "75"] = data.T["score"].apply(
                temp_fuc, args=(test, "end")
            )
            college_df.rename(columns={(test + "25"): test_name + "25%"}, inplace=True)
            college_df.rename(columns={(test + "75"): test_name + "75%"}, inplace=True)
    elif college_info.data_type == "学生组成":
        data = pd.read_json(dataURLs["students"] + str(college_info.postid))
        df1 = (
            data[["name", "value"]]
            .rename(columns={"name": "学生类别", "value": "数量"})
            .replace(
                {
                    "uf": lang_dict["data_under_fresh"],
                    "uj": lang_dict["data_under_junior"],
                    "ut": lang_dict["data_under_trans"],
                    "gr": lang_dict["data_under_grad"],
                    "nd": lang_dict["data_no_degree"],
                }
            )
        )
        tempArray = {
            "wh": lang_dict["data_race_white"],
            "as": lang_dict["data_race_asian"],
            "la": lang_dict["data_race_latino"],
            "pa": lang_dict["data_race_pacific"],
            "af": lang_dict["data_race_africa"],
            "nr": lang_dict["data_race_nr"],
        }
        df2 = pd.DataFrame(columns=["学生种族", "数量"])

        def temp_fuc(row, k):
            for subrow in row:
                if subrow["name"] == k:
                    return subrow["value"]

        for key, value in tempArray.items():
            temp = pd.DataFrame(
                {
                    "学生种族": [value],
                    "数量": [data["subs"].apply(temp_fuc, args=([key])).sum()],
                }
            )
            df2 = pd.concat([df2, temp], ignore_index=True)
        college_df = {str(int(data["year"][0])) + "年学生组成": [df1, df2]}
    elif college_info.data_type == "学生人数":
        college_df = pd.read_json(
            dataURLs["students_number"] + str(college_info.postid)
        )
        college_df = college_df.rename(
            columns={
                "undertotal": "本科生人数",
                "under": "本科留学生人数",
                "underper": "本科留学生占比",
                "gradtotal": "研究生人数",
                "grad": "研究生留学生人数",
                "gradper": "研究生留学生占比",
            }
        )
        college_df["year"] = college_df["year"].apply(lambda x: x[:-1])
    elif college_info.data_type == "学费":
        data = pd.read_json(dataURLs["info"] + str(college_info.postid))
        college_df = pd.DataFrame()
        tuition_array = {
            "year": "year",
            "tuition_in_under": "州内本科生学费",
            "tuition_out_under": "外州本科生学费",
            "tuition_in_grad": "州内研究生学费",
            "tuition_out_grad": "外州研究生学费",
            "room": "住宿费",
        }
        for key, value in tuition_array.items():
            college_df[value] = data["tuition"].apply(lambda x: x[key])
        college_df["year"] = college_df["year"].astype(str)
    elif college_info.data_type == "毕业率":
        data = pd.read_json(dataURLs["info"] + str(college_info.postid))
        college_df = pd.DataFrame()
        college_df["year"] = data["graduation"].apply(lambda x: x["year"])
        college_df["毕业率"] = data["graduation"].apply(
            lambda x: x["graduation_100_under"]
        )
        college_df["学生保有率"] = data["retention"].apply(
            lambda x: x["retention_under"]
        )
    elif college_info.data_type == "犯罪率":
        college_df = pd.read_json(dataURLs["crime"] + str(college_info.postid))
        college_df = college_df[["year", "avg1000"]].rename(
            columns={"avg1000": "每千人学生记过、犯罪率"}
        )
    return {"college_info": college_info, "question": question, "data": college_df}


# Plot College Data
def plot_college_data(df, data_type):
    """Plot the college data."""
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        fig, ax = plt.subplots(figsize=(9, 4))
        if data_type == "排名":
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 8))
            axes[0].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[0].transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            axes[0].plot(df["USNews排名"], "o-")
            axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].invert_yaxis()
            axes[0].set_ylabel(lang_dict["data_ranking"])
            axes[1].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[1].transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            world_ranks = ["QS", "USNews", "THE", "ARWU"]
            for world_rank in world_ranks:
                axes[1].plot(
                    df[world_rank + "世界大学排名"],
                    "o-",
                    label=world_rank + " " + lang_dict["data_world_ranking"],
                )
            axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1].invert_yaxis()
            axes[1].set_ylabel(lang_dict["data_world_ranking"])
            axes[1].legend()
        elif data_type == "录取率":
            ax.text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=ax.transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            ax.plot(df.index, df["男生录取率"], "bo-", label=lang_dict["data_men_adm"])
            ax.plot(
                df.index, df["女生录取率"], "ro-", label=lang_dict["data_women_adm"]
            )
            ax.set_ylabel(lang_dict["data_admission_rate"])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            plt.legend()
        elif data_type == "申请录取人数":
            ax.text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=ax.transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            lns1 = ax.plot(
                df.index,
                df["录取且入学人数"],
                "bo-",
                label=lang_dict["data_enroll_num"],
            )
            lns2 = ax.plot(
                df.index,
                df["录取但未入学人数"],
                "ro-",
                label=lang_dict["data_defer_num"],
            )
            ax_twin = ax.twinx()
            lns3 = ax_twin.plot(
                df.index, df["拒绝人数"], "go-", label=lang_dict["data_reject_num"]
            )
            ax.set_ylabel(lang_dict["data_enrollment"])
            ax_twin.set_ylabel(lang_dict["data_reject_num"])
            lns = lns1 + lns2 + lns3
            labs = [ln.get_label() for ln in lns]
            plt.legend(lns, labs, loc=0)
        elif data_type == "成绩要求":
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 10))
            if df["SAT阅读25%"].isnull().sum() < df["SAT阅读25%"].shape[0]:
                axes[0, 0].fill_between(df.index, df["SAT阅读25%"], df["SAT阅读75%"])
            axes[0, 0].set_ylabel(lang_dict["data_sat_reading"])
            if df["SAT数学25%"].isnull().sum() < df["SAT数学25%"].shape[0]:
                axes[0, 1].fill_between(df.index, df["SAT数学25%"], df["SAT数学75%"])
            axes[0, 1].set_ylabel(lang_dict["data_sat_math"])
            if df["ACT英语25%"].isnull().sum() < df["ACT英语25%"].shape[0]:
                axes[1, 0].fill_between(
                    df.index,
                    df["ACT英语25%"].dropna(),
                    df["ACT英语75%"].dropna(),
                    color="coral",
                )
            axes[1, 0].set_ylabel(lang_dict["data_act_english"])
            if df["ACT数学25%"].isnull().sum() < df["ACT数学25%"].shape[0]:
                axes[1, 1].fill_between(
                    df.index,
                    df["ACT数学25%"].dropna(),
                    df["ACT数学75%"].dropna(),
                    color="coral",
                )
            axes[1, 1].set_ylabel(lang_dict["data_act_math"])
            if df["ACT综合25%"].isnull().sum() < df["ACT综合25%"].shape[0]:
                axes[2, 0].fill_between(
                    df.index,
                    df["ACT综合25%"].dropna(),
                    df["ACT综合75%"].dropna(),
                    color="coral",
                )
            axes[2, 0].set_ylabel(lang_dict["data_act_comp"])
            axes[2, 1].axis("off")
        elif data_type == "学生组成":
            df = next(iter(df.values()))
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))

            def my_autopct(pct):
                return ("%.1f%%" % pct) if pct > 5 else ""

            explode = (
                (df[0]["学生类别"] == lang_dict["data_under_fresh"]) / 6
            ).to_list()
            axes[0].pie(
                df[0]["数量"],
                labels=df[0]["学生类别"],
                autopct=my_autopct,
                explode=explode,
            )
            explode = ((df[1]["学生种族"] == lang_dict["data_race_nr"]) / 6).to_list()
            axes[1].pie(
                df[1]["数量"],
                labels=df[1]["学生种族"],
                explode=explode,
                autopct=my_autopct,
            )
        elif data_type == "学生人数":
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 8))
            axes[0].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[0].transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            axes[0].plot(
                df["year"],
                df["本科生人数"],
                "o-",
                label=lang_dict["data_under_students_num"],
            )
            axes[0].plot(
                df["year"],
                df["研究生人数"],
                "o-",
                label=lang_dict["data_grad_students_num"],
            )
            axes[0].set_ylabel(lang_dict["data_students_num"])
            axes[0].tick_params(axis="x", rotation=30)
            axes[0].legend()
            axes[1].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[1].transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            axes[1].plot(
                df["year"],
                df["本科留学生人数"],
                "o-",
                label=lang_dict["data_under_nr_num"],
            )
            axes[1].plot(
                df["year"],
                df["研究生留学生人数"],
                "o-",
                label=lang_dict["data_grad_nr_num"],
            )
            axes[1].set_ylabel(lang_dict["data_international_students_number"])
            axes[1].tick_params(axis="x", rotation=30)
            axes[1].legend()
            plt.tight_layout()
        elif data_type == "学费":
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
            axes[0, 0].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[0, 0].transAxes,
                fontsize=20,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=30,
            )
            axes[0, 0].plot(
                df["year"],
                df["州内本科生学费"],
                "o-",
                label=lang_dict["data_tuition_in_under"],
            )
            axes[0, 0].plot(
                df["year"],
                df["州内研究生学费"],
                "o-",
                label=lang_dict["data_tuition_in_grad"],
            )
            axes[0, 0].set_ylabel(lang_dict["data_tuition_fees_in"])
            axes[0, 0].tick_params(axis="x", rotation=30)
            axes[0, 0].legend()
            axes[0, 1].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[0, 1].transAxes,
                fontsize=20,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=30,
            )
            axes[0, 1].plot(
                df["year"],
                df["外州本科生学费"],
                "o-",
                label=lang_dict["data_tuition_out_under"],
            )
            axes[0, 1].plot(
                df["year"],
                df["外州研究生学费"],
                "o-",
                label=lang_dict["data_tuition_out_grad"],
            )
            axes[0, 1].set_ylabel(lang_dict["data_tuition_fees_out"])
            axes[0, 1].tick_params(axis="x", rotation=30)
            axes[0, 1].legend()
            axes[1, 0].text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=axes[1, 0].transAxes,
                fontsize=20,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=30,
            )
            axes[1, 0].plot(
                df["year"], df["住宿费"], "o-", label=lang_dict["data_room_board"]
            )
            axes[1, 0].set_ylabel(lang_dict["data_room_board"])
            axes[1, 0].tick_params(axis="x", rotation=30)
            axes[1, 0].legend()
            axes[1, 1].axis("off")
            plt.tight_layout()
        elif data_type == "毕业率":
            ax.text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=ax.transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            ax.plot(
                df["year"], df["毕业率"], "o-", label=lang_dict["data_graduation_rate"]
            )
            ax.plot(
                df["year"],
                df["学生保有率"],
                "o-",
                label=lang_dict["data_retention_rate"],
            )
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend()
        elif data_type == "犯罪率":
            ax.text(
                0.5,
                0.5,
                "Forward Pathway",
                transform=ax.transAxes,
                fontsize=50,
                color="gray",
                alpha=0.02,
                ha="center",
                va="center",
                rotation=20,
            )
            ax.plot(
                df["year"],
                df["每千人学生记过、犯罪率"],
                "o-",
                label=lang_dict["data_crime_rate"],
            )
            ax.set_ylabel(lang_dict["data_crime_rate"])
        st.pyplot(fig)


def college_data_comments(state: GraphState):
    """Generate the comments for the college data."""
    df = state["data"]
    data_type = state["college_info"].data_type
    college_cname = state["college_info"].cname
    college_ename = state["college_info"].ename
    college_url = "https://www.forwardpathway.com/" + state["college_info"].postid
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", lang_dict["prompt_comments_system"]),
            ("human", lang_dict["prompt_comments_human"]),
        ]
    )
    college_data_comments_chain = prompt | llm | StrOutputParser()
    generation = college_data_comments_chain.stream(
        {
            "college_cname": college_cname,
            "college_ename": college_ename,
            "data_type": data_type,
            "data": df,
            "college_url": college_url,
            "question": question,
        }
    )
    return {"generation": generation}


def generate_retrieve_question(state: GraphState):
    """Reroute the user's question from database to the RAG."""
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "基于用户的问题，重新生成一个可以更好查询vector store以取得相关内容文章的短语",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": question})
    return {"question": new_question}


def ranking_data(state: GraphState):
    """Get the ranking data from the database."""
    question = state["question"]
    chat_history = state["chat_history"]
    available_types = CollegeRanking.get_ranking_types()

    structured_llm = llm.with_structured_output(RankingType, method="json_schema")
    system_message = """你是一名熟悉美国大学排名的专家，下面将给出用户的一个问题和历史聊天记录，你需要按照以下步骤判断用户的问题是列表中的哪一种排名类型。\
        1. 如果用户提问的学院或者专业大类在列表的school栏中存在，则输出该列表的这行数据。
        2. 如果用户提问的专业属于列表中school栏中某一学院下属专业，则输出该列表的这行的数据。
        2. 如果用户提问的学院或者专业大类在列表的school栏中不存在，且专业也不属于school栏下属专业的，则school输出NULL，level输出本科，year输出NULL。
        3. 如果用户提问的是美国大学排名，不涉及任何学院或者专业的，则school输出NULL，level输出本科，year输出NULL。"""
    llm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "human",
                "用户问题如下：{question}\n\n历史聊天记录如下：{chat_history}\n\n可以选择的排名类型有：{available_types}",
            ),
        ]
    )
    question_router = llm_prompt | structured_llm
    ranking_type = question_router.invoke(
        {
            "question": question,
            "chat_history": chat_history,
            "available_types": available_types,
        }
    )
    if ranking_type.school and ranking_type.school != "NULL":
        ranking_type_str = ranking_type.school + ranking_type.level + "排名"
        ranking_year = ranking_type.year
        ranking_df = CollegeRanking.get_major_ranking(ranking_type)
    else:
        ranking_type_str = "美国大学排名"
        (ranking_year, ranking_df) = CollegeRanking.get_usnews_ranking()

    return {
        "ranking_df": ranking_df,
        "ranking_year": ranking_year,
        "ranking_type": ranking_type_str,
    }


def ranking_output(state: GraphState):
    """Generate the ranking response."""
    ranking_df = state["ranking_df"]
    ranking_year = state["ranking_year"]
    ranking_type = state["ranking_type"]
    question = state["question"]
    chat_history = state["chat_history"]

    # Create a prompt template for generating the ranking response
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", lang_dict["prompt_ranking_system"]),
            ("human", lang_dict["prompt_ranking_human"]),
        ]
    )

    # Chain the prompt with the language model and output parser
    ranking_output_chain = prompt | llm | StrOutputParser()

    # Invoke the chain to generate the ranking response
    ranking_response = ranking_output_chain.stream(
        {
            "question": question,
            "ranking_year": ranking_year,
            "ranking_type": ranking_type,
            "ranking_df": ranking_df,
            "chat_history": chat_history,
        }
    )

    return {"generation": ranking_response}


# Build LangGraph
workflow = StateGraph(GraphState)

workflow.add_node("router_college", router_college)
workflow.add_node("router_ranking", router_ranking)
workflow.add_node("retrieve", retrieve)
workflow.add_node("get_college_info", get_college_info)
workflow.add_node("generate", generate)
workflow.add_node("college_data_plot", college_data_plot)
workflow.add_node("college_data_comments", college_data_comments)
workflow.add_node("generate_retrieve_question", generate_retrieve_question)
workflow.add_node("ranking_data", ranking_data)
workflow.add_node("ranking_output", ranking_output)

workflow.add_edge(START, "router_college")
workflow.add_conditional_edges(
    "router_college",
    router_college_func,
    {
        "to_database": "get_college_info",
        "to_router_ranking": "router_ranking",
    },
)
workflow.add_conditional_edges(
    "router_ranking",
    router_ranking_func,
    {
        "to_retrieve": "generate_retrieve_question",
        "to_ranking": "ranking_data",
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

workflow.add_conditional_edges(
    "get_college_info",
    database_router_func,
    {
        "to_college_data_plot": "college_data_plot",
        "to_retrieve": "generate_retrieve_question",
    },
)

workflow.add_edge("generate_retrieve_question", "retrieve")
workflow.add_edge("college_data_plot", "college_data_comments")
workflow.add_edge("college_data_comments", END)
workflow.add_edge("ranking_data", "ranking_output")
workflow.add_edge("ranking_output", END)
app = workflow.compile()


def draw_graph_png():
    """Draw the graph of the LangChain."""
    from langchain_core.runnables.graph import (CurveStyle, MermaidDrawMethod,
                                                NodeStyles)

    img = app.get_graph(xray=1).draw_mermaid_png(
        curve_style=CurveStyle.BASIS,
        node_colors=NodeStyles(
            first="fill:#FDFFB6",
            last="fill:#FFADAD",
            default="fill:#CAFFBF,line-height:1",
        ),
        draw_method=MermaidDrawMethod.API,
    )
    with open("GraphFlow.png", "wb") as png:
        png.write(img)


# draw_graph_png()
# Build Streamlit APP
with st.sidebar:
    st.subheader(lang_dict["more"], divider="rainbow")
    lang_dict["rankings"]
    lang_dict["lxbd"]
    lang_dict["service_under"]
    lang_dict["service_grad"]
    lang_dict["service_emergency"]
    st.divider()
    st.subheader(lang_dict["service_barcode"])
    st.image("./logos/WeCom_barcode.png", width=200)
    st.divider()
    st.markdown(lang_dict["disclaim"])
st.title(lang_dict["title"])

avatars = {"assistant": "./logos/fp_logo.png", "user": "❓"}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": lang_dict["init_content"]}
    ]
if "input_time" not in st.session_state:
    st.session_state.input_time = datetime.datetime.now() - datetime.timedelta(
        seconds=10
    )

# print messages
for msg in st.session_state.messages:
    if msg["role"] == "user" or msg["role"] == "assistant":
        with st.chat_message(msg["role"], avatar=avatars[msg["role"]]):
            st.write(msg["content"])
    elif msg["role"] == "data":
        plot_college_data(msg["content"], msg["data_type"])
if user_input := st.chat_input(lang_dict["input_box"]):
    if (datetime.datetime.now() - st.session_state.input_time) < datetime.timedelta(
        seconds=3
    ):
        st.error(lang_dict["error_too_many_requests"])
        st.snow()
    else:
        st.session_state.input_time = datetime.datetime.now()
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=avatars["user"]):
            st.write(user_input)

        inputs = {
            "question": user_input,
            "chat_history": st.session_state.messages[-10:],
        }
        placeholder = st.empty()
        with placeholder.container():
            status = st.status(lang_dict["status_wait"])
        try:
            for output in app.stream(inputs):
                for key, response in output.items():
                    if "generation" in response:
                        status.update(label=lang_dict["status_generate"])
                        msg = response["generation"]
                        with st.chat_message("assistant", avatar=avatars["assistant"]):
                            msg = st.write_stream(msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": msg}
                            )
                    elif "data" in response:
                        status.update(label=lang_dict["status_generate"])
                        data = response["data"]
                        data_type = (response["college_info"]).data_type
                        plot_college_data(data, data_type)
                        st.session_state.messages.append(
                            {"role": "data", "content": data, "data_type": data_type}
                        )
        except Exception as e:
            logger.info(user_input)
            logger.error(e)
            with st.chat_message("assistant", avatar=avatars["assistant"]):
                st.write(lang_dict["error"])
        status.update(label=lang_dict["status_finish"], state="complete")
        time.sleep(1)
        placeholder.empty()
