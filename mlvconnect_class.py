import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusClient, db
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

class DBConnection:
    def __init__(self, db_name="default", collection_name="joaquin_DB", uri_connection="http://localhost:19530", host="localhost", port=19530, dimension_embedding=3072):
        self.uri_connection = uri_connection
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.dimension_embedding = dimension_embedding
        self.api_key_openAI = None
        self.client = None
        self.vector_store = None

    def load_api_key(self):
        load_dotenv(find_dotenv(), override=True)
        self.api_key_openAI = os.environ.get("OPENAI_API_KEY")
        if not self.api_key_openAI:
            raise ValueError("OPENAI_API_KEY no encontrado en el archivo .env")
        print(self.api_key_openAI)

    def connect_database(self, ):
        connections.connect(
            uri=self.uri_connection,
            token="joaquin:chamorro",
            db_name=self.db_name,
            alias="default"
        )
        self.client = MilvusClient(uri=self.uri_connection, token="joaquin:chamorro")
        self.client.using_database(self.db_name, using="default")
        print(f"Conectado a la base de datos {self.db_name}")
        
    def vectors_store(self):   
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.api_key_openAI)
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64}
        }

        self.vector_store = Milvus(
            embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.uri_connection},
            enable_dynamic_field=True,
            primary_field="pk",
            text_field="text",
            vector_field="vector",
            index_params=index_params
        )

        print(f"Conexión a Milvus-VectorStore establecida.\nConectado a coleccion: {self.collection_name}\n")
        
    def process(self):
        self.load_api_key()
        self.connect_database()
        self.vectors_store()
        


# Uso de ejemplo (descomentar y reemplazar con tus valores)
# uri_connection = 'your_uri_connection'
# host = 'your_host'
# port = 'your_port'
# db_name = 'your_db_name'
# collection_name = 'your_collection_name'
# dimension_embedding = 3072  # Ejemplo de dimensión

# processor = DBConnection(uri_connection, host, port, db_name, collection_name, dimension_embedding)
# processor.process()
