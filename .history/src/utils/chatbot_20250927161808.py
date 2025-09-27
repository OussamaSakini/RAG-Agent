import time
from openai import OpenAI
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict
from utils.config import AppConfig

URL = "https://github.com/OussamaSakini/RAG-Agent"
hyperlink = f"[RAG-GPT guideline]({URL})"

app_config = AppConfig()

class ChatBot:
    
    @staticmethod
    def response(chatbot:List[Dict], message:str, type:str = "Preprocessed doc"):
        embedding = OllamaEmbeddings(model=app_config.embedding_model, base_url="http://localhost:11434")
        
        if type == "Preprocessed doc":
            persist_directory = app_config.persist_directory
            if os.path.exists(app_config.persist_directory):
                vectordb = Chroma(persist_directory = persist_directory, 
                            embedding_function = embedding)
            else:
                chatbot.append(
                    {"role" : "assistant", "content" : f"VectorDB does not exist. Please first upload the data manually. For further information please visit {hyperlink}."})
                return "", chatbot, None
            
        elif type == "Upload doc: Process for RAG":
            custom_directory = app_config.custom_persist_directory
            if os.path.exists(custom_directory):
                vectordb = Chroma(persist_directory = custom_directory, 
                            embedding_function = embedding)
            else:
                chatbot.append(
                    {"role" : "assistant", "content" : f"No file was uploaded. Please first upload your file(s) using the 'upload' button."})
                return "", chatbot, None
            
        docs = vectordb.similarity_search(message, k=app_config.k)
        refs_md_lines = [
            f"[{i+1}] {d.page_content}\n"
            f"Source: {d.metadata.get('source')} p.{d.metadata.get('page')}"
            for i, d in enumerate(docs)
        ]
        
        retrieved_content = "### Retrieved content\n\n" + "\n\n".join(refs_md_lines)
        
        question = "# User new question:\n" + message
        
        # Memory: previous four Q&A pairs
        N = app_config.number_of_memory
        
        if 1 <= len(chatbot) <= 3:
            pairs = chatbot[:]   
        elif len(chatbot) >= 4:
            pairs = chatbot[-N:]
        else:
            pairs = []

        if not pairs:
            chat_history = "Chat history:\n (vide pour l'instant)\n\n"
        else:
            history_lines = [f"{pair["role"]} : {pair["content"]}" for pair in pairs]
            chat_history = "Chat history:\n" + "\n\n".join(history_lines) + "\n\n"
            
        prompt = f"{chat_history}\n{retrieved_content}\n{question}"
        print("========================")
        print("User question : ",message)
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=app_config.llm_model,
            messages=[
                {"role": "system", "content": app_config.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        chatbot.append({"role" : "user", "content" : message})
        chatbot.append({"role" : "assistant", "content" : response.choices[0].message.content})
        time.sleep(2)

        return "", chatbot, , "salam"
