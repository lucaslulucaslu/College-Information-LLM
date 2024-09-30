"""This file contains the language dictionaries for the chatbot."""
cn_dict = {
    "prompt_retriever": "基于以上聊天内容及用户最新的问题，生成一个独立的可以用来查找与问题相关内容的查询句子，句子尽量包含所有细节",
    "prompt_document": """你是美国续航教育的问答小助手，基于之前的聊天记录，并尽量使用下面提供的美国续航教育资料库内容生成问题的答案，\
        不要捏造答案，如果回答中有提到帮助或者服务的内容，尽可能在回答最后推荐使用美国续航教育的相关服务并给出可以点击的美国续航教育官网url：\
            https://www.forwardpathway.com，美国续航教育资料库内容如下：\n\n{context}\n\n聊天记录如下：{chat_history}""",
    "prompt_comments_system": "你是一位了解美国高等院校各类数据的专家，你需要对下面这所大学的该类数据做一个简短的讨论，\
        可以是趋势、变化、异常或者其他任何与该数据相关的情况,如果是录取率，保有率，毕业率等小于1的数值，请自动转换成百分比数值，\
            数据来源url：{college_url}，请给出一个可以点击的正确链接地址，链接显示名为该大学名称。\
                如果数据以及讨论没有很好的回答用户的问题，请进行必要的补充，用户原始问题如下：{question}",
    "prompt_comments_human": "大学名称：{college_cname}，数据类型：{data_type}，具体数据如下：\n\n{data}",
    "prompt_ranking_system": """基于下面的排名数据和用户的问题，生成一个回答，回答要求如下：
                1. 生成的表明表格提供排名，中文名，英文名三栏，
                2. 如无特殊要求的，生成排名表格只输出前10名的数据。
                3. 生成回答时的排名类型、排名年份请按照提供的排名数据类型和排名年份为准，只会有美国大学排名和学院排名，不会细分到专业排名。
                4. 如果提供的排名数据类型位美国大学排名，则生成美国大学排名表格，\
                    在表格后要告诉用户细分项类别的排名请参考美国续航教育的美国大学排名页面：https://www.forwardpathway.com/ranking。
                5. 如果提供的排名数据类型为学院排名，而用户提问的排名类型位专业排名，则生成学院排名表格，\
                    在表格后要告诉用户细分项类别的排名请参考美国续航教育的美国大学排名页面：https://www.forwardpathway.com/ranking。
                6. 如果提供的排名数据类型为学院排名，而用户提问的排名类型同样是学院排名，则生成学院排名表格。\
                    在表格后要告诉用户其他类别的排名请参考美国续航教育的美国大学排名页面：https://www.forwardpathway.com/ranking。
                \n\n排名年份：{ranking_year}\n\n排名类型：{ranking_type}\
                    \n\n排名数据如下：{ranking_df}""",
    "prompt_ranking_human": "用户问题如下：{question}\n\n历史聊天记录如下：{chat_history}",
    "title": "💬 美国续航教育AI小助手",
    "init_content": "这里是美国续航教育AI小助手，请问有什么可以帮您的吗？",
    "input_box": "试试：哈佛大学的录取率是多少？",
    "more": "更多资源可点击链接查看",
    "rankings": "[美国大学排名数据库](https://www.forwardpathway.com/ranking)",
    "lxbd": "[留学宝典](https://www.forwardpathway.com/usabaike)",
    "service_under": "[美国大学申请服务](https://www.forwardpathway.com/university-application)",
    "service_grad": "[研究生、博士申请服务](https://www.forwardpathway.com/graduate-apply)",
    "service_emergency": "[留学紧急情况应对服务](https://www.forwardpathway.com/emergency-transfer)",
    "service_barcode": "微信扫码联系在线客服",
    "data_ranking": "USNews排名",
    "data_world_ranking": "世界大学排名",
    "data_admission_rate": "录取率",
    "data_men_adm": "男生录取率",
    "data_women_adm": "女生录取率",
    "data_enrollment": "录取、拒绝人数",
    "data_enroll_num": "录取且入学人数",
    "data_defer_num": "录取但未入学人数",
    "data_reject_num": "拒绝人数",
    "data_sat_reading": "SAT阅读",
    "data_sat_math": "SAT数学",
    "data_act_comp": "ACT综合",
    "data_act_english": "ACT英语",
    "data_act_math": "ACT数学",
    "data_under_fresh": "本科新生",
    "data_under_junior": "本科老生",
    "data_under_trans": "本科转学生",
    "data_under_grad": "研究生",
    "data_no_degree": "无学位",
    "data_race_white": "白人",
    "data_race_asian": "亚裔",
    "data_race_latino": "拉丁裔",
    "data_race_pacific": "太平洋岛国",
    "data_race_africa": "非裔",
    "data_race_nr": "留学生",
    "data_under_students_num": "本科生人数",
    "data_grad_students_num": "研究生人数",
    "data_under_nr_num": "本科留学生人数",
    "data_grad_nr_num": "研究生留学生人数",
    "data_students_num": "学生数量",
    "data_international_students_number": "留学生数量",
    "data_tuition_in_under": "州内本科生学费",
    "data_tuition_out_under": "外州本科生学费",
    "data_tuition_in_grad": "州内研究生学费",
    "data_tuition_out_grad": "外州研究生学费",
    "data_room_board": "住宿、生活费",
    "data_tuition_fees_in": "州内学生学费",
    "data_tuition_fees_out": "外州学生学费",
    "data_graduation_rate": "毕业率",
    "data_retention_rate": "学生保有率",
    "data_crime_rate": "每千人学生记过、犯罪率",
    "error_too_many_requests": "为了AI服务器更好的运行，两次提问输入间隔请不要少于3秒时间！",
    "languages": "语言选择",
    "disclaim": "*该APP资料及数据来源为美国续航教育官网，输出内容经ChatGPT整理，APP测试阶段回答不一定准确，请确认后使用",
    "status_wait": "整理资料中，请稍候……",
    "status_generate": "输出资料中……",
    "status_finish": "完成。",
    "error": "抱歉，出现内部问题，请换个问法试试！",
}
en_dict = {
    "prompt_retriever": "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
    "prompt_document": """Consider chat history and answer the user's questions based on the below context:\n\n{context}\n\n \
        chat history: {chat_history}\n\nIf there is any consulting service or related agents are mentioned in the answer, \
            please recommend Forward Pathway's services and give a clickable company website url:https://www.forwardpathway.com""",
    "prompt_comments_system": """You are a expert for U.S. colleges' data, based on the data below, you need to comment on the data, \
        comments can be trends, anormaly, change or anything related to the data provided, if data is related to admission rate, \
            retention rate or graduation rate and numbers are below 1, you need to convert the number to percentage number automatically, \
                data source url: {college_url}，please cite a clickable link at the end of comments""",
    "prompt_comments_human": "college name: {college_ename}，data type: {data_type}，detailed data:\n\n{data}",
    "prompt_ranking_system": """Based on the following ranking data and the user's question, \
                    generate a response with the following requirements:
                1. The generated table should provide two columns: Rank, and English Name.
                2. Unless otherwise specified, the generated ranking table should only output the top 10 entries.
                3. When generating the response, the ranking type and ranking year should be based on the provided \
                    ranking data type and ranking year. Only US university/college rankings and school rankings will be provided, \
                        not detailed to specific majors.
                4. If the provided ranking data type is US university rankings, generate a US university ranking table.\
                    After the table, inform the user that for detailed category rankings, they should refer to the \
                        Forward Pathway US university ranking page: https://www.forwardpathway.us/ranking.
                5. If the provided ranking data type is college/school rankings, and the user's question is about major rankings, \
                    generate a college/school ranking table. After the table, inform the user that for detailed category rankings, \
                        they should refer to the Forward Pathway US university ranking page: https://www.forwardpathway.us/ranking.
                6. If the provided ranking data type is college/school rankings, and the user's question is also about college rankings, \
                    generate a college ranking table. After the table, inform the user that for other category rankings, \
                        they should refer to the Forward Pathway US university ranking page: https://www.forwardpathway.us/ranking.
                \n\nRanking Year: {ranking_year}\n\nRanking Type: {ranking_type}\n\nRanking Data: {ranking_df}""",
    "prompt_ranking_human": "User question: {question}\n\nChat history: {chat_history}",
    "title": "💬 Forward Pathway AI ChatBot",
    "init_content": "How can I help you?",
    "input_box": "Try: What's UCLA's rankings?",
    "more": "More resources",
    "rankings": "[College Rankings](https://www.forwardpathway.com/ranking)",
    "lxbd": "[International Students Handbook](https://www.forwardpathway.com/usabaike)",
    "service_under": "[Under Application Services](https://www.forwardpathway.com/university-application)",
    "service_grad": "[Grad/PHD Application Services](https://www.forwardpathway.com/graduate-apply)",
    "service_emergency": "[Emergency Services](https://www.forwardpathway.com/emergency-transfer)",
    "service_barcode": "WeChat barcode for agents",
    "data_ranking": "USNews Rank",
    "data_world_ranking": "World Us Rank",
    "data_admission_rate": "Admission Rate",
    "data_men_adm": "Men Admission Rate",
    "data_women_adm": "Women Admisson Rate",
    "data_enrollment": "Enroll/Reject Number",
    "data_enroll_num": "Enroll",
    "data_defer_num": "Defer",
    "data_reject_num": "Reject",
    "data_sat_reading": "SAT Reading",
    "data_sat_math": "SAT Math",
    "data_act_comp": "ACT Composite",
    "data_act_english": "ACT English",
    "data_act_math": "ACT Math",
    "data_under_fresh": "Freshmen",
    "data_under_junior": "Continuing",
    "data_under_trans": "Transfer-ins",
    "data_under_grad": "Graduate",
    "data_no_degree": "Non-degree",
    "data_race_white": "White",
    "data_race_asian": "Asian",
    "data_race_latino": "Hispanic",
    "data_race_pacific": "Pacific & others",
    "data_race_africa": "African American",
    "data_race_nr": "International",
    "data_under_students_num": "Under Students",
    "data_grad_students_num": "Grad Students",
    "data_under_nr_num": "Under International",
    "data_grad_nr_num": "Grad International",
    "data_students_num": "Students Number",
    "data_international_students_number": "International Students",
    "data_tuition_in_under": "Under in-state tuition",
    "data_tuition_out_under": "Under out-of-state tuition",
    "data_tuition_in_grad": "Grad in-state tuition",
    "data_tuition_out_grad": "Grad out-of-state tuition",
    "data_room_board": "Room & Board",
    "data_tuition_fees_in": "In-state Tuition",
    "data_tuition_fees_out": "Out-of-state Tuition",
    "data_graduation_rate": "Graduation Rate",
    "data_retention_rate": "Retention Rate",
    "data_crime_rate": "Crime rate per 1000 students",
    "error_too_many_requests": "Please wait for 3 seconds to input another question!",
    "disclaim": "*The data and information of this APP are from the official website of Forward Pathway LLC, \
        and the output content has been organized by ChatGPT. \
            Please note that the answers during the testing phase of the APP may not be accurate, so please verify before using.",
    "status_wait": "Putting everything togerth...",
    "status_generate": "Generating content...",
    "status_finish": "Finished.",
    "error": "Sorry, there is an internal error, please try another question!",
}
