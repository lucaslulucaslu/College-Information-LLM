"""This module defines the data models schema."""

from typing import List, Literal, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field


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

    school: Optional[str] = Field(description="学院名称、专业大类名称")
    level: Literal["本科", "研究生"] = Field(description="本科、研究生")
    year: Optional[int] = Field(description="排名年份")


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
    router_college_flag: str
    router_ranking_flag: str
    generation: str
    documents: List[str]
    college_info: College_Info
    ranking_year: int
    ranking_type: str
    ranking_df: pd.DataFrame
    data_type: str
    plot_type: str
    data: pd.DataFrame
    chat_history: list


class CollegeRouter(BaseModel):
    """基于用户的查询词条选择是否为单所大学相关问题."""

    college: Literal["Yes", "No"] = Field(
        description="回答是否为某所特定大学相关问题，Yes为某所大学相关问题，No为不相关"
    )


class RouteQuery(BaseModel):
    """基于用户的查询词条选择最相关的资料来源，vectorstore或者ranking."""

    router: Literal["vectorstore", "ranking"] = Field(
        description="基于用户的问题选择vectrostore或者ranking。"
    )
