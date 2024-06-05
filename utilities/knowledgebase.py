from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

CHUNK_SIZE=4000
CHUNK_OVERLAP=400
HIDE_SOURCE_DOCUMENTS=False

class TXTKnowledgeBase:
    def __init__(self,txt_source_folder_path:str)->None:
        self.txt_source_folder_path=txt_source_folder_path
        
    def load_txts(self):
        loader=DirectoryLoader(
            self.txt_source_folder_path
        )
        loaded_txts=loader.load()
        return loaded_txts
        
    def split_documents(self,loaded_docs):
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunked_docs=text_splitter.split_documents(loaded_docs)
        return chunked_docs
        
    def convert_documents_to_embeddings(self,chunked_docs):
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small',chunk_size=CHUNK_SIZE)
        vector=FAISS.from_documents(chunked_docs,embeddings)
        vector.save_local(
            folder_path='vector',
            index_name=('knowledge-base-vector-'+self.txt_source_folder_path)
        )
        return vector

    def return_retriever_from_persistant_vector_db(self):
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small',chunk_size=CHUNK_SIZE)
        vector=FAISS.load_local(
            folder_path='vector',
            index_name=('knowledge-base-vector-'+self.txt_source_folder_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector
        
    def initiate_documents(self):
        loaded_txts=self.load_txts()
        chunked_documents=self.split_documents(loaded_docs=loaded_txts)
        print("=> TXT files loading and chunking done.")
        vector=self.convert_documents_to_embeddings(chunked_docs=chunked_documents)
        print("=> vector initialised")
        return vector
    def initiate_documents_nochunk(self):
        loaded_txts=self.load_txts()
        #chunked_documents=self.split_documents(loaded_docs=loaded_txts)
        #print("=> TXT files loading and chunking done.")
        vector=self.convert_documents_to_embeddings(chunked_docs=loaded_txts)
        print("=> vector initialised")
        return vector