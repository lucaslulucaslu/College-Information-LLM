"""Forward Pathway AI Chatbot."""

import datetime
import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langgraph.graph import END, START, StateGraph
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, PercentFormatter

from utilities import languages
from utilities.colleges import CollegesData
from utilities.knowledgebase import TXTKnowledgeBase
from utilities.ranking import CollegeRanking
from utilities.schema import (
    College_Info,
    MainRouter,
    GraphState,
    RankingType,
)
from utilities.llm_wrapper import llm_wrapper, llm_wrapper_streaming
from langfuse.decorators import observe


logger = logging.getLogger(__name__)
logging.basicConfig(filename="error.log", encoding="utf-8", level=logging.ERROR)

st.set_page_config(
    page_title="Forward Pathway AI Chatbot",
    page_icon="./logos/fp_logo.png",
    menu_items={
        "About": "APP资料及数据来源为美国续航教育官网，输出内容经ChatGPT整理，APP测试阶段回答不一定准确，请确认后使用"
    },
)

SEARCH_DOCS_NUM = 8
SEARCH_COLLEGES_NUM = 3


@st.cache_resource
def get_combined_retriever():
    """只在启动时加载并合并一次向量库"""
    logger.info("Initializing Vector Databases...")
    kb_lxbd = TXTKnowledgeBase(
        txt_source_folder_path="lxbd"
    ).return_retriever_from_persistant_vector_db()
    kb_lxsq = TXTKnowledgeBase(
        txt_source_folder_path="lxsq"
    ).return_retriever_from_persistant_vector_db()
    kb_emergency = TXTKnowledgeBase(
        txt_source_folder_path="emergency"
    ).return_retriever_from_persistant_vector_db()

    # 预先合并
    kb_lxbd.merge_from(kb_lxsq)
    kb_lxbd.merge_from(kb_emergency)

    # 返回预设好的 retriever
    return kb_lxbd.as_retriever(search_kwargs={"k": SEARCH_DOCS_NUM})


get_combined_retriever()  # 预加载向量数据库

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


@observe
def unified_router(state: GraphState):
    """一键分发用户意图到三个核心路径。"""

    system_prompt = """你是一名熟悉美国大学申请的资深顾问。请根据用户的问题和历史对话，将其路由到最合适的处理路径：

1. **college_info (特定学校路径)**: 
   - 触发条件：问题核心是关于“某一所特定大学”的具体数据或信息。
   - 例子：'哈佛大学的排名'、'普林斯顿的录取率'、'CMU的学费'、'去UIUC安全吗'。
   - 注意：只要问题聚焦在单所学校，即使涉及排名也选此项。

2. **ranking (排名列表路径)**: 
   - 触发条件：用户想要查看某类学校的“排名榜单”或“对比列表”，不针对单一特定学校。
   - 例子：'美国大学排名'、'商科专业排名'、'最好的工程学院有哪些'、'加州有哪些好的公立大学'。

3. **vectorstore (综合知识库路径)**: 
   - 触发条件：关于留学流程、政策、多校综合对比或泛指的申请建议。
   - 例子：'美国大学申请流程'、'如何转学到美国'、'会计专业就业前景'、'托福要考多少分'、'哈佛和耶鲁哪个好'。

请务必精准判断。"""

    user_prompt = f"当前问题：{state['question']}\n\n历史对话：{state['chat_history']}"

    # 调用 LLM 进行解析
    response = llm_wrapper(
        system_prompt, user_prompt, response_format=MainRouter
    ).parsed

    return {"router_flag": response.route}


def unified_router_func(state: GraphState):
    """LangGraph 路由逻辑跳转控制"""
    route = state["router_flag"]
    if route == "college_info":
        return "to_database"
    elif route == "ranking":
        return "to_ranking"
    else:
        return "to_retrieve"


@observe
def retrieve(state: GraphState):
    retriever = get_combined_retriever()  # 这里获取的是缓存好的对象，速度极快
    retrieval_query = state["retrieval_query"]
    documents = retriever.invoke(retrieval_query)
    return {"documents": documents}


@observe
def generate(state: GraphState):
    """Generate the answer to the user's question."""
    documents = state["documents"]

    system_prompt = lang_dict["prompt_document"].format(
        context=documents, chat_history=state["chat_history"]
    )
    if "college_info" in state:
        college_info = {
            "中文名": state["college_info"].cname,
            "英文名": state["college_info"].ename,
            "url": "https://www.forwardpathway.com/" + state["college_info"].postid,
        }
        system_prompt += f"\n\n问题中提到的学校信息如下：{college_info}， 可以推荐查看该学校的url，用下面的markdown格式输出可以点击的url:[{state["college_info"].cname}]({college_info["url"]})以获取更详细的学校信息。"
    user_prompt = "用户的问题如下：" + state["question"]
    response = llm_wrapper_streaming(system_prompt, user_prompt)

    return {"generation": response}


@observe
def get_college_info(state: GraphState):
    """Get the college information from the database."""
    # print("---COLLEGE NAME---")

    college_data = CollegesData()
    college_vector = college_data.return_colleges_vector_from_db()
    college_retriever = college_vector.as_retriever(
        search_kwargs={"k": SEARCH_COLLEGES_NUM}
    )
    system_prompt = "你是一位了解美国高等院校的专家，你需要根据用户的问题及历史聊天记录提取出一所美国高等院校的全名，包括中文名和英文名，输出格式为'中文全名（英文全名）'"
    user_prompt = f"用户问题如下：{state["question"]}\n\n历史聊天记录如下：{state["chat_history"]}"
    college_name = llm_wrapper(system_prompt, user_prompt).text
    college_context = college_retriever.invoke(college_name)

    system_prompt = (
        "基于下面学校信息内容及用户的问题和历史聊天记录，按照格式输出学校信息回答"
    )
    user_prompt = f"用户问题如下：{state["question"]}\n\n学校信息内容如下：{college_context}\n\n历史聊天记录如下：{state["chat_history"]}"
    respone = llm_wrapper(
        system_prompt, user_prompt, response_format=College_Info
    ).parsed

    return {"college_info": respone}


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
        and post_id.isdigit()
    ):
        return "to_college_data_plot"
    else:
        return "to_retrieve"


@observe
def college_data_plot(state: GraphState):
    """Get the college data from the database."""
    question = state["question"]
    college_info = state["college_info"]
    dataURLs = {
        "rank_adm": "https://www.forwardpathway.com/d3v7/dataphp/school_database/ranking_admin_20250923.php?name=",
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


@observe
def college_data_comments(state: GraphState):
    """Generate the comments for the college data."""
    df = state["data"]
    data_type = state["college_info"].data_type
    college_cname = state["college_info"].cname
    college_ename = state["college_info"].ename
    college_url = "https://www.forwardpathway.com/" + state["college_info"].postid
    question = state["question"]
    system_prompt = lang_dict["prompt_comments_system"].format(
        college_url=college_url, question=question
    )
    user_prompt = lang_dict["prompt_comments_human"].format(
        college_cname=college_cname,
        college_ename=college_ename,
        data_type=data_type,
        data=df,
    )
    response = llm_wrapper_streaming(system_prompt, user_prompt)
    return {"generation": response}


@observe
def generate_retrieve_question(state: GraphState):
    """Reroute the user's question from database to the RAG."""
    system_prompt = "基于用户的问题和历史聊天记录，重新生成一个可以更好查询vector store以取得相关内容文章的短语。注意：最终只输出一个你认为最合适的搜索短语。"
    user_prompt = f"用户问题如下：{state["question"]}，\n\n历史聊天记录如下：{state["chat_history"]}"
    response = llm_wrapper(system_prompt, user_prompt).text
    return {"retrieval_query": response}


@observe
def ranking_data(state: GraphState):
    """Get the ranking data from the database."""
    question = state["question"]
    chat_history = state["chat_history"]
    available_types = CollegeRanking.get_ranking_types()

    system_prompt = """你是一名熟悉美国大学排名的专家，下面将给出用户的一个问题和历史聊天记录，结合用户问题和历史聊天记录，了解用户问题的真实意图。\
        你需要按照以下步骤判断用户问题的意图是列表中的哪一种排名类型。\
        1. 如果用户问题的意图的学院或者专业大类在列表的school栏中存在，则输出该列表的这行数据。
        2. 如果用户问题的意图的专业属于列表中school栏中某一学院下属专业，则输出该列表的这行的数据。
        2. 如果用户问题的意图的学院或者专业大类在列表的school栏中不存在，且专业也不属于school栏下属专业的，则school输出NULL，level输出本科，year输出NULL。
        3. 如果用户问题的意图是美国大学排名，不涉及任何学院或者专业的，则school输出NULL，level输出本科，year输出NULL。"""
    user_prompt = f"用户问题如下：{question}\n\n历史聊天记录如下：{chat_history}\n\n可以选择的排名类型有：{available_types}"
    ranking_type = llm_wrapper(
        system_prompt, user_prompt, response_format=RankingType
    ).parsed
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


@observe
def ranking_output(state: GraphState):
    """Generate the ranking response."""
    ranking_df = state["ranking_df"]
    ranking_year = state["ranking_year"]
    ranking_type = state["ranking_type"]
    question = state["question"]
    chat_history = state["chat_history"]

    system_prompt = lang_dict["prompt_ranking_system"].format(
        ranking_year=ranking_year, ranking_type=ranking_type, ranking_df=ranking_df
    )
    user_prompt = lang_dict["prompt_ranking_human"].format(
        question=question, chat_history=chat_history
    )
    # Create a prompt template for generating the ranking response
    response = llm_wrapper_streaming(system_prompt, user_prompt)
    return {"generation": response}


# --- 构建节点 ---
workflow = StateGraph(GraphState)

# 注册所有节点
workflow.add_node("unified_router", unified_router)
workflow.add_node("get_college_info", get_college_info)
workflow.add_node("ranking_data", ranking_data)
workflow.add_node("generate_retrieve_question", generate_retrieve_question)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("college_data_plot", college_data_plot)
workflow.add_node("college_data_comments", college_data_comments)
workflow.add_node("ranking_output", ranking_output)

# --- 构建逻辑边界 ---
workflow.add_edge(START, "unified_router")

workflow.add_conditional_edges(
    "unified_router",
    unified_router_func,
    {
        "to_database": "get_college_info",  # 去查特定学校数据库
        "to_ranking": "ranking_data",  # 去查排名榜单
        "to_retrieve": "generate_retrieve_question",  # 去查 RAG 知识库
    },
)

workflow.add_conditional_edges(
    "get_college_info",
    database_router_func,
    {
        "to_college_data_plot": "college_data_plot",
        "to_retrieve": "generate_retrieve_question",
    },
)

workflow.add_edge("generate_retrieve_question", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

workflow.add_edge("college_data_plot", "college_data_comments")
workflow.add_edge("college_data_comments", END)

workflow.add_edge("ranking_data", "ranking_output")
workflow.add_edge("ranking_output", END)

app = workflow.compile()


@observe(name="Chatbot")
def langgraph_app_stream(input):
    return app.stream(input)


def draw_graph_png():
    """Draw the graph of the LangChain."""
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

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
            for output in langgraph_app_stream(inputs):
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
