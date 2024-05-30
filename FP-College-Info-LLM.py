import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from knowledgebase import TXTKnowledgeBase
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "college-information-llm"

SEARCH_DOCS_NUM=2

llm = ChatOpenAI()
kb=TXTKnowledgeBase(txt_source_folder_path='lxbd')
#kb.initiate_documents()
vector=kb.return_retriever_from_persistant_vector_db()

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever(search_kwargs={'k':SEARCH_DOCS_NUM})

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions only based on the below context:\n\n{context}\n\n 如果回答中有提到帮助或者服务的项目，尽可能在回答最后推荐使用美国续航教育的相关服务"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history=''

def stream_response():
    response_stream=retrieval_chain.stream({
            'chat_history':chat_history,
            'input':prompt
        })
    for r in response_stream:
        if 'answer' in r:
            yield r['answer']
st.markdown(
        r"""<style>
    .viewerBadge_container__r5tak {
            visibility: hidden;
        }</style>
    """, unsafe_allow_html=True
    )                
with st.sidebar:
    st.subheader('更多资源可点击链接查看',divider='rainbow')
    "[美国大学排名数据库](https://www.forwardpathway.com/ranking)"
    "[留学宝典](https://www.forwardpathway.com/usabaike)"
    "[美国大学申请服务](https://www.forwardpathway.com/university-application)"
    "[研究生、博士申请服务](https://www.forwardpathway.com/graduate-apply)"
    "[留学紧急情况应对服务](https://www.forwardpathway.com/emergency-transfer)"
    st.divider()
    st.subheader("微信扫码联系在线客服")
    st.image('./logos/WeCom_barcode.png')

st.title("💬 美国续航教育AI小助手")

avatars={'assistant':'./logos/fp_logo.png','user':'❓'}

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "这里是美国续航教育AI小助手，请问有什么可以帮您的吗？"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"],avatar=avatars[msg['role']]):
        st.markdown(msg["content"])

if prompt := st.chat_input('请输入您的问题，问题请尽量详细。'):
    if not os.environ['OPENAI_API_KEY']:
        st.info("Please add your OpenAI API key in ENV to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user",avatar=avatars['user']):
        st.markdown(prompt)
        
    chat_history=st.session_state.messages
    
    with st.chat_message("assistant",avatar=avatars['assistant']):
        msg=st.write_stream(stream_response)
        
    st.session_state.messages.append({"role": "assistant", "content": msg})
    