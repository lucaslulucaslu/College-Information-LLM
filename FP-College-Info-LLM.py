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
from langchain_core.messages import HumanMessage, AIMessage
import languages

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "college-information-llm"

##############choose language######################
lan_index=('lan' in st.query_params and st.query_params['lan']=='EN')

def change_language_fuc():
    del st.session_state["messages"]

language=st.radio(
    label='Language:',
    options=['CN','EN'],
    horizontal=True,
    index=lan_index,
    on_change=change_language_fuc
)
if language=='EN':
    lang_dict=languages.en_dict
else:
    lang_dict=languages.cn_dict

###################
    
SEARCH_DOCS_NUM=2

llm = ChatOpenAI(model='gpt-4o')
kb=TXTKnowledgeBase(txt_source_folder_path='lxbd')
#kb.initiate_documents()
vector=kb.return_retriever_from_persistant_vector_db()

retriever = vector.as_retriever(search_kwargs={'k':SEARCH_DOCS_NUM})

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", lang_dict['prompt_retriever'])
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", lang_dict['prompt_document']),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history=[]

def stream_response():
    response_stream=retrieval_chain.stream({
            'chat_history':chat_history,
            'input':prompt
        })
    for r in response_stream:
        if 'answer' in r:
            yield r['answer']

def chat_history_generater(msgs):
    chat_history=[]
    for msg in msgs:
        if msg['role']=='user':
            chat_history.append(HumanMessage(content=msg['content']))
        if msg['role']=='assistant':
            chat_history.append(AIMessage(content=msg['content']))
    return chat_history

with st.sidebar:
    st.subheader(lang_dict['more'],divider='rainbow')
    lang_dict['rankings']
    lang_dict['lxbd']
    lang_dict['service_under']
    lang_dict['service_grad']
    lang_dict['service_emergency']
    st.divider()
    st.subheader(lang_dict['service_barcode'])
    st.image('./logos/WeCom_barcode.png')


st.title(lang_dict['title'])


avatars={'assistant':'./logos/fp_logo.png','user':'❓'}

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": lang_dict['init_content']}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"],avatar=avatars[msg['role']]):
        st.markdown(msg["content"])

if prompt := st.chat_input(lang_dict['input_box']):
    
    chat_history=chat_history_generater(st.session_state.messages)
    
    if not os.environ['OPENAI_API_KEY']:
        st.info("Please add your OpenAI API key in ENV to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user",avatar=avatars['user']):
        st.markdown(prompt)
    
    with st.chat_message("assistant",avatar=avatars['assistant']):
        msg=st.write_stream(stream_response)
        
    st.session_state.messages.append({"role": "assistant", "content": msg})
    