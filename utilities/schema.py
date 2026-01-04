"""This module defines the data models schema."""

from typing import Literal, TypedDict

import pandas as pd
from pydantic import BaseModel, Field


class MainRouter(BaseModel):
    """根据用户意图选择最佳处理路径。"""

    route: Literal["college_info", "ranking", "vectorstore"] = Field(
        ...,
        description="选择处理路径：'college_info' 针对特定某一所学校的问题；'ranking' 针对排名列表类问题；'vectorstore' 针对通用留学知识或多校对比。",
    )


class College_Info(BaseModel):
    """Represents the college information."""

    cname: str = Field(description="学校中文全名")
    ename: str = Field(description="学校英文全名")
    postid: str = Field(description="学校的postid")
    unitid: str = Field(description="学校的unitid")
    data_type: str = Field(
        description="数据种类，可以是'排名'、'录取率'、'申请录取人数'、'成绩要求'、'学生组成'、'学生人数'、'学费'、'毕业率'、'犯罪率'这几种中的一种，\
            如果留学生相关则输出'学生人数'，如果涉及住宿费则输出'学费'，如果涉及学生保有率则输出'毕业率'，如果不在以上这些类型中请输出'不是数据'"
    )


class RankingType(BaseModel):
    """Represents the ranking type."""

    school: str | None = Field(description="学院名称、专业大类名称")
    level: Literal["本科", "研究生"] = Field(description="本科、研究生")
    year: int | None = Field(description="排名年份")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        college_name: College name
        data_type: College data type
        plot_type: College data plot type, can be line plot, bar plot, scatter plot or tree plot
    """

    question: str
    retrieval_query: str
    router_flag: str
    generation: str
    documents: list[str]
    college_info: College_Info
    ranking_year: int
    ranking_type: str
    ranking_df: pd.DataFrame
    data_type: str
    plot_type: str
    data: pd.DataFrame
    chat_history: list
