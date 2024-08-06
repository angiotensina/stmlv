import os
from dotenv import load_dotenv, find_dotenv
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient
import nest_asyncio

# import LlamaParse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# import Langchain modules
from langchain.agents import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus.vectorstores import Milvus

######################### VARIABLES DE ENTORNO #########################
QUERY = "¿Qué es el Parkinson?"

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSION_EMBEDDING = 3072

INPUT_FILE = "parkinson_gpt.txt"
BD_NAME = "Colono_db"
COLLECTION_NAME = "Colono_col"

URI_CONNECTION = "http://localhost:19530"
HOST = "localhost"
PORT = 19530

ARCHIVO_PDF = "./PK/CURRENT NEURO PARKINSON.pdf"
ARCHIVO_SALIDA = "./PK/CURRENT NEURO PARKINSON.txt"

######################### CARGAR EL DOCUMENTO #########################
def chunks_from_txtfile(file):
   # load the document and split it into chunks

    # loader = PyPDFDirectoryLoader("./folder")
    loader = TextLoader(file)
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n", 
        chunk_overlap=0
        )
    docs = text_splitter.split_documents(documents)

    ######################## ADDING METADATA TO DOCUMENTS ########################
    # Adding metadata to documents
    for i, doc in enumerate(docs):
        doc.metadata["chunk"] = i
        doc.metadata["Title"] = "Parkinson"
        doc.metadata["Capitulo"] = "30"
        doc.metadata["author"] = "Joaquin Chamorro"
    
    return docs


def embed_documents_to_milvus(txt, uri, db_name, collection_name):
    ######################### OBTENER API KEY DE OPENAI #########################
    # Carga las variables de entorno desde un archivo .env
    load_dotenv(find_dotenv(), override=True)

    # Obtiene la API key de OpenAI desde las variables de entorno
    api_key_openAI = os.environ.get("OPENAI_API_KEY")
    print(api_key_openAI)

    docs = chunks_from_txtfile(txt)

    ######################### CONECTAR A MILVUS #########################

    _, hostv, portv = uri.split(":")
    hostv = str(hostv[2:])
    portv = int(portv)
    
    client = MilvusClient(
        uri=uri,
        token="joaquin:chamorro"
    )

    connections.connect(alias="default", host=hostv, port=portv)

    ######################### CREAR LA BASE DE DATOS EN MILVUS #########################

    
    if db_name not in db.list_database():
        db.create_database(db_name, using="default")
        db.using_database(db_name, using="default")
    else:
        db.using_database(db_name, using="default")

    print(f"Conectado a la base de datos {db_name}")

    
    ######################### GUARDAR LOS VECTORES EN MILVUS - VECTORSTORE #########################
    #vector_store = Milvus()

    # Crear la función de embeddings de código abierto
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key_openAI)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64}
    }

    if collection_name not in client.list_collections():
        vector_store = Milvus.from_documents(
            docs,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={"uri": uri},
            primary_field="pk",
            text_field="text",
            vector_field="vector",
            index_params=index_params,
            enable_dynamic_field=True,
            drop_old=True
        )
        print("Colección creada")
    else:
        vector_store = Milvus(
            embeddings,
            collection_name=collection_name,
            connection_args={"uri": uri},
            enable_dynamic_field=True,
            primary_field="pk",
            text_field="text",
            vector_field="vector",
            index_params=index_params
        )
        print("Colección ya existe")

    print(f"Conexión a Milvus-VectorStore establecida.\nConectado a colleccion: {collection_name}\n")
    os.remove(txt)
    return vector_store

def parse_document(archivo_pdf):
    nest_asyncio.apply()

    # Carga las variables de entorno desde un archivo .env
    load_dotenv(find_dotenv(), override=True)

    # Obtiene la API key de OpenAI desde las variables de entorno
    llama_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    parser = LlamaParse(
        api_key= llama_key,
        result_type="text",  # "markdown" and "text" are available
        parsing_instruction=
        """
        - Elimina encabezados y pies de pagina de cada pagina del PDF.
        - Organiza tablas en lineas de texto.
        - organiza todo el texto en una sola pagina separada por parrafos.
        - Los párrafos no finalizados complétalo con la información existente en la página siguiente, o en el párrafo siguiente.
        - Organiza en párrafos separados por dos retornos de carro.
        - Traduce el texto al español.
        """,
        language="es",
        skip_diagonal_text= False,
        invalidate_cache= False,
        gpt4o_mode= False,
        gpt4o_api_key= openai_key,
        verbose=True,
        show_progress=True,
    )

    # sync
    documents = parser.load_data(archivo_pdf)

    # sync batch
    #documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])

    # async
    #documents = await parser.aload_data("./my_file.pdf")

    # async batch
    #documents = await parser.aload_data(["./my_file1.pdf", "./my_file2.pdf"])

    # Extraer texto de un directorio
    """
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        "./PK", file_extractor=file_extractor
    ).load_data()
    """
    #print(documents[0].text)
    filename = "test.txt"
    
    if documents:  # Verifica si la lista 'documents' no está vacía
        with open(filename, "w") as f:
            f.write(documents[0].text)
    else:
        print("No se encontraron documentos.")
    
    return filename

def main():
    # Cargar el documento
    file = "./tst.pdf"
    
    #fileTOembed = "./colonoscopy.txt"
    #filename=fileTOembed
    
    filename=parse_document(file)
    vectorstore = embed_documents_to_milvus(txt=filename, uri="http://localhost:19530", db_name="tst_db", collection_name="tst_col")
    print("Documento cargado en Milvus")
    
if __name__ == "__main__":
    main()