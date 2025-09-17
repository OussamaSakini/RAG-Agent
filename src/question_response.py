from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
import yaml
from openai import OpenAI

with open("config/app_config.yml") as cfg:
    app_config = yaml.load(cfg, yaml.FullLoader)
    
embedding = OllamaEmbeddings(model=app_config["embedding_model_config"]["engine"], base_url="http://localhost:11434")
vectordb = Chroma(
    persist_directory= app_config["directories"]["persist_directory"],
    embedding_function =embedding
)
print("Number of vectors in vectordb:", vectordb._collection.count())


while True:
    question_user = input("Enter your question or type \"q\" t quit")
    if question_user.lower() =='q':
        break
    question = "# User question:\n" + question_user
    docs_retrived_response = vectordb.similarity_search(question, app_config["retrieval_config"]["k"])
    retrived_response: List[str] = [f"{str(page.page_content)} \n" 
        f"Source: {page.metadata.get('source')}"
        for page in docs_retrived_response]
    retrived = "# Retrieved content number:\n" + str(retrived_response)
    prompt = retrived + "\n\n" + question
    
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.chat.completions.create(
        model=app_config["llm_config"]["engine"],
        messages=[
            {"role" : "system", "content": app_config["llm_config"]["llm_system_role"]},
            {"role" : "user", "content": prompt}
        ]
    )
    
    print(response.choices[0].message.content)