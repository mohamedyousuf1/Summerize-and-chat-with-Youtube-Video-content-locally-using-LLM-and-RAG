from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from utils import *

class RAG:
    def __init__(self):
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"}, cache_folder="instructor_cache")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""] )

        self.persist_directory = 'YT_video' # directory to save the embeddings vectors

    def process_YT_url(self,video_URL, local_llm):
        
        self.loader = YoutubeLoader.from_youtube_url(
            video_URL, add_video_info=False
        )
        self.documents = self.loader.load()

        self.texts = self.text_splitter.split_documents(self.documents)
        self.vectordb = Chroma.from_documents(documents=self.texts,
                                 embedding=self.instructor_embeddings,
                                 persist_directory=self.persist_directory)
        
        
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

        self.qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                  chain_type="stuff",
                                  retriever=self.retriever,
                                  return_source_documents=True)
                
    def make_query(self, query):
        self.llm_response = self.qa_chain(query)

        return process_llm_response(self.llm_response)
    
    def make_query_GUI(self, query):
        self.llm_response = self.qa_chain(query)
        # metadata_list = []
        # for source in self.llm_response["source_documents"]:
        #     print(source.metadata['source'])
        #     metadata_list.append(source.metadata['source'])
        return self.llm_response['result']

    
