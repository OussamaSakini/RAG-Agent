
from utils.prepare_vectordb import PrepareVectorDB
from typing import List, Tuple
from utils.config import AppConfig

app_config = AppConfig()

class UploadFile:
    """
    Utility class for handling file uploads and processing.

    This class provides static methods for checking directories and processing uploaded files
    to prepare a VectorDB.
    """
    
    @staticmethod
    def uploads_files(files_dir: List, chatbot: List, btn_upload : str):
        """
        Process uploaded files to prepare a VectorDB.

        Parameters:
            files_dir (List): List of paths to the uploaded files.
            chatbot: An instance of the chatbot for communication.
        """
        if btn_upload == "Upload doc: Process for RAG":
            preparator = PrepareVectorDB(
                files_dir,
                app_config.custom_persist_directory,
                app_config.chunk_size,
                app_config.chunk_overlap,
                app_config.embedding_model
            )
            
            preparator.prepare_and_save_vectorDB()
            
            chatbot.append({"role": "assistant", "content": "Upload completed successfully."})
        else:
            chatbot.append({"role": "assistant", "content": "Please choose the option \"Upload doc: Process for RAG\" to upload files."})

        return "", chatbot

