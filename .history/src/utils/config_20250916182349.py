import yaml

class AppConfig:
    def __init__(self) -> None:
        with open("config/app_config.yml") as cfg:
            app_config = yaml.load(cfg, yaml.FullLoader)
        
        # Directories
        self.data_directory = app_config["directories"]["data_directory"]
        self.persist_directory = app_config["directories"]["persist_directory"]
        self.custom_persist_directory = app_config["directories"]["custom_persist_directory"]
        
        # embedding_model_config
        self.embedding_model = app_config["embedding_model_config"]["engine"]
        
        # LLM Config
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.llm_model = app_config["llm_config"]["engine"]
        
        # Splitter config
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]
        
        # Retrieval config
        self.k = app_config["retrieval_config"]["k"]
        
        # Memory
        self.number_of_memory = app_config["memory"]["number_of_memory"]