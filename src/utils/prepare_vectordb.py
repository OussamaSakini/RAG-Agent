from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List
import os
from pyprojroot import here
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml


class PrepareVectorDB:
    """
    Build and persist a vector database from documents.

    This class loads documents, splits them into overlapping chunks, computes
    text embeddings with the chosen model, and writes the resulting index to
    `persist_directory`.

    Args:
        files_directory (str | list[str]): Path to a folder (or list of folders) containing the documents to index.
        persist_directory (str): Directory where the vector DB will be stored.
        chunk_size (int): Target size of each text chunk.
        chunk_overlap (int): Number of characters (or tokens) overlapping between consecutive chunks to preserve context.
        text_embedding_model (str): Name of the embedding model.
    """
    
    def __init__(self, files_directory, persist_directory, chunk_size, chunk_overlap, text_embedding_model) -> None:
        self.files_directory = files_directory
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_embedding_model = text_embedding_model
        self.embedding = OllamaEmbeddings(model=self.text_embedding_model, base_url="http://localhost:11434")
        
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def __load_documents(self) -> List:
        """
        Load all documents from the specified directory or directories.

        Returns:
            List: A list of loaded documents.
        """
        docs = []
        counter_docs = 0
                
        if isinstance(self.files_directory, list):
            for file_path in self.files_directory:
                if os.path.isfile(file_path):
                    docs.extend(PyPDFLoader(file_path).load())
                    counter_docs += 1
        elif isinstance(self.files_directory, list):            
            base = here(self.files_directory)
            if os.path.exists(base) :
                all_files = os.listdir(base)
                for doc in all_files:
                    file_path = os.path.join(base,doc) 
                    if os.path.isfile(file_path):
                        docs.extend(PyPDFLoader(file_path).load())
                        counter_docs += 1
        else:
            print(f"Directory {base} doesn't exist")
            
        print("Files loaded : ", counter_docs)
        print("Total pages : ",len(docs))
        
        return docs
    
    def __chunks_documents(self, docs:list) :
        """
        Chunk the loaded documents using the specified text splitter.

        Parameters:
            docs (List): The list of loaded documents.

        Returns:
            List: A list of chunked documents.

        """
        
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks : ", len(chunked_documents))
        return chunked_documents
    
    
    def prepare_and_save_vectorDB(self) :
        """
        Load, chunk, and create a VectorDB with OpenAI embeddings, and save it.

        Returns:
            Chroma: The created VectorDB.
        """
        
        documents = self.__load_documents()
        if not documents:
            raise RuntimeError("No documents loaded. Check files_directory and PDF files.")
    
        documents_chunked = self.__chunks_documents(documents)
        
        os.makedirs((self.persist_directory), exist_ok= True)
        vectordb = Chroma.from_documents(
            persist_directory = self.persist_directory,
            embedding = self.embedding,
            documents = documents_chunked
        )
        
        print("VectorDB is created and saved.")
        
        return vectordb
    

if __name__ == "__main__":
    with open(here("RAG AGENT/config/app_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        
    initialisation = PrepareVectorDB(
        files_directory = app_config["directories"]["data_directory"],
        persist_directory = app_config["directories"]["persist_directory"],
        chunk_size = app_config["splitter_config"]["chunk_size"],
        chunk_overlap = app_config["splitter_config"]["chunk_overlap"],
        text_embedding_model = app_config["embedding_model_config"]["engine"],
    )
    
    initialisation.prepare_and_save_vectorDB()