cn_dict={
    'template1':"""尽量仅使用下面提供的内容生成问题的答案:

<context>
{context}
</context>

问题: {input}""",
    'template2':'基于以上聊天内容及用户最新的问题，生成一个独立的可以用来查找与问题相关内容的查询短语',
    'template3':"""你是美国续航教育的问答小助手，尽量仅使用下面提供的内容生成问题的答案，如果不知道答案请回答不知道，不要捏造答案:\n\n{context}\n\n 如果回答中有提到帮助或者服务的内容，尽可能在回答最后推荐使用美国续航教育的相关服务""",
    'title':'💬 美国续航教育AI小助手',
    'init_content':'这里是美国续航教育AI小助手，请问有什么可以帮您的吗？',
    'input_box':'请输入您的问题，问题请尽量详细。',
    'more':'更多资源可点击链接查看',
    'rankings':'[美国大学排名数据库](https://www.forwardpathway.com/ranking)',
    'lxbd':"[留学宝典](https://www.forwardpathway.com/usabaike)",
    'service_under':"[美国大学申请服务](https://www.forwardpathway.com/university-application)",
    'service_grad':"[研究生、博士申请服务](https://www.forwardpathway.com/graduate-apply)",
    'service_emergency':"[留学紧急情况应对服务](https://www.forwardpathway.com/emergency-transfer)",
    'service_barcode':'微信扫码联系在线客服'
}
en_dict={
    'template1':"""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""",
    'template2':'Given the above conversation, generate a search query to look up to get information relevant to the conversation',
    'template3':"""Answer the user's questions only based on the below context:\n\n{context}""",
    'title':'💬 Forward Pathway AI ChatBot',
    'init_content':'How can I help you?',
    'input_box':'Please input your question',
    'more':'More resources',
    'rankings':'[College Rankings](https://www.forwardpathway.com/ranking)',
    'lxbd':"[International Students Handbook](https://www.forwardpathway.com/usabaike)",
    'service_under':"[Under Application Services](https://www.forwardpathway.com/university-application)",
    'service_grad':"[Grad/PHD Application Services](https://www.forwardpathway.com/graduate-apply)",
    'service_emergency':"[Emergency Services](https://www.forwardpathway.com/emergency-transfer)",
    'service_barcode':'WeChat barcode for agents'
}