'''
This file contains the RAG class which is used to process the youtube video URL and make queries to the model.

'''

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from utils import *

class RAG:
    def __init__(self):
        '''
        The RAG class contains the following methods:
            __init__: This method is used to initialize the RAG class.
            process_YT_url: This method is used to process the youtube video URL.
            make_query: This method is used to make a query to the model.
            make_query_GUI: This method is used to make a query to the model and return the result.

        Attributes:
            instructor_embeddings: The instructor embeddings are loaded from the HuggingFaceInstructEmbeddings model.
            text_splitter: The text splitter is loaded from the RecursiveCharacterTextSplitter model.
            persist_directory: The persist directory is set to 'YT_video'.
            loader: The loader is initialized with the YoutubeLoader model.
            documents: The documents are loaded from the loader.
            texts: The texts are split using the text splitter.
            vectordb: The vectordb is loaded from the Chroma model.
            retriever: The retriever is loaded from the vectordb.
            qa_chain: The qa_chain is loaded from the RetrievalQA model.

        Methods:
            process_YT_url: This method is used to process the youtube video URL.
            make_query: This method is used to make a query to the model.
            make_query_GUI: This method is used to make a query to the model and return the result.
        '''

        # Load the instructor embeddings, which are used to embed the texts
        # embedding means converting the text into a vector representation
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"}, cache_folder="instructor_cache")
        
        # Load the text splitter, which is used to split the texts into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""] )
        
        self.persist_directory = 'YT_video' # directory to save the embeddings vectors

    def process_YT_url(self,video_URL, local_llm):
        '''
        This method is used to process the youtube video URL.
        Args:
            video_URL: The video URL.
            local_llm: The local_llm.
        Returns:
            The texts that are split using the text splitter.

        Steps:
            - Load the instructor embeddings.
            - Load the text splitter.
            - Set the persist directory to 'YT_video'.
            - Initialize the loader with the YoutubeLoader model.
            - Load the documents from the loader.
            - Split the texts using the text splitter.
            - Load the vectordb from the Chroma model.
            - Load the retriever from the vectordb.
            - Load the qa_chain from the RetrievalQA
        '''
        
        self.loader = YoutubeLoader.from_youtube_url(
            video_URL, add_video_info=False
        )
        # Load the instructor embeddings, 
        self.documents = self.loader.load()

        # Split the texts using the text splitter, the texts are split into chunks of 4000 characters with 50 characters overlap
        self.texts = self.text_splitter.split_documents(self.documents)

        # Load the vectordb from the Chroma model, vectordb is used to store the embeddings of the texts 
        self.vectordb = Chroma.from_documents(documents=self.texts,
                                 embedding=self.instructor_embeddings,
                                 persist_directory=self.persist_directory)
        
        # Load the retriever from the vectordb, retriever is used to retrieve similar documents
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3}) # k=3 is the number of similar documents to retrieve

        # Load the qa_chain from the RetrievalQA, qa_chain is used to make queries to the model
        self.qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                  chain_type="stuff",
                                  retriever=self.retriever,
                                  return_source_documents=True)
        return self.texts
                
    def make_query(self, query):
        '''
        This method is used to make a query to the model.
        Args:
            query: The query.
        Returns:
            The processed llm response.
        '''
        self.llm_response = self.qa_chain(query)

        return process_llm_response(self.llm_response)
    
    def make_query_GUI(self, query):
        '''
        This method is used to make a query to the model and return the result.
        Args:
            query: The query.
        Returns:
            The result of the query.
        '''
        self.llm_response = self.qa_chain(query)
        return self.llm_response['result']

    
