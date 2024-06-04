from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

CHUNK_SIZE=200

class CollegesData:
    def __init__(self)->None:
        pass
        
    def initial_colleges_vector_to_db(self):
        colleges_df=pd.read_csv('.\\colleges\\colleges.csv')
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small')
        row_txt_format='中文名：{c}，英文名：{e}，类型：{t}，postid：{p}，unitid：{u}，所在州：{s}'
        college_types={1:'综合大学',2:'文理学院',3:'社区大学'}
        txts=[]
        for index,row in colleges_df.iterrows():
            row_txt=row_txt_format.format(c=row['cname'],e=row['name'],t=college_types[row['type']],p=row['postid'],u=row['unitid'],s=row['state'])
            txts.append(row_txt)
        vectors=FAISS.from_texts(txts,embeddings)
        vectors.save_local(folder_path='vector',index_name='colleges-data-vector')
        return vectors

    def return_colleges_vector_from_db(self):
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small')
        vectors=FAISS.load_local(
            folder_path='vector',
            index_name='colleges-data-vector',
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vectors