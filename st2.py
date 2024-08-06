import streamlit as st
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient
import os
from dotenv import load_dotenv, find_dotenv
from embed_class import EmbeddingProcessor
from parse_class import PDFParser
from mlvconnect_class import DBConnection

######################### VARIABLES DE ENTORNO #########################
QUERY = "Tecnicas para evitar la formacion de asas durante la colonoscopia"

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSION_EMBEDDING = 3072

INPUT_FILE = "./colonoscopy.txt"
OUTPUT_FILE = f"{INPUT_FILE}_output.txt"
BD_NAME = "RAG"
COLLECTION_NAME = "Colonoscopy"

URI_CONNECTION = "http://localhost:19530"
HOST = "localhost"
PORT = 19530

def embeddings_function(uploaded_file, db_name, collection_name):
    archivo_pdf = './colonoscopy.txt'
    uri_connection = 'http://localhost:19530'
    host = 'localhost'
    port = '19530'
    dimension_embedding = 3072  # Example dimension

    #archivo_pdf = "./tst.pdf"
    archivo_salida = "./tst.txt"
    st.write(f"Archivo PDF: {uploaded_file.name}")
    st.write("Parseando PDF...")
    parser = PDFParser(uploaded_file.name, archivo_salida)
    parser.parse_pdf()
    st.write("Parseo completado.")
    st.write("Creando embeddings...")
    processor = EmbeddingProcessor(parser.archivo_salida, uri_connection, host, port, db_name, collection_name, dimension_embedding)
    processor.process()
    st.write("Embeddings creados.")
    st.write("Conexión a Milvus-VectorStore establecida.")
    st.write("Proceso completado.")
    #os.remove(parser.archivo_salida)

def getcollection(vector_index):
    client = MilvusClient(uri=URI_CONNECTION)
    client.using_database(vector_index, using="default")
    return client.list_collections()

def main():
    # Cabecera de la app
    col1, col2 = st.columns(2)
    col1.selectbox('IP server', options=['BERT', 'RoBERTa', 'DistilBERT'])
    col2.selectbox('Port', options=['BERT', 'RoBERTa', 'DistilBERT'])
    st.markdown("<H1 style='text-align: center'> RAG PROGRAM </H1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("<H1 style='text-align: center'> Panel principal </H1>", unsafe_allow_html=True)
    radioselected = st.sidebar.radio('Selecciona una opción', ['Embeddings', 'RAG', 'Index'])

    # Main panel
    if radioselected == 'Embeddings':
        with st.form(key='embeddings_form', clear_on_submit=True):
            st.write('Embeddings')
            uploaded_file = st.file_uploader("Elige un archivo PDF", type=["pdf"], accept_multiple_files=False)
            if uploaded_file is not None:
                st.write("filename:", uploaded_file.name)
            
            st.multiselect('Select columns', ['col1', 'col2', 'col3'])
            col1a, col2a = st.columns(2)
            db_name = col1a.text_input('Enter database')
            collection_name = col2a.text_input('Enter collection')
            
            submitted = st.form_submit_button('Create Embeddings')
            
            if submitted:
                embeddings_function(uploaded_file, db_name, collection_name)

    if radioselected == 'RAG':
        connections.connect(
            uri="http://localhost:19530",
            token="joaquin:chamorro",
            alias="default"
        )

        # Obtener lista de bases de datos
        list_databases = db.list_database()

        # Inicializar la sesión de Streamlit
        if 'db_select' not in st.session_state:
            st.session_state.db_select = list_databases[0]
        if 'collections' not in st.session_state:
            st.session_state.collections = getcollection(st.session_state.db_select)

        # Seleccionar base de datos fuera del formulario
        vector_index = st.selectbox(
            'Select database',
            list_databases,
            key='db_select',
            on_change=lambda: st.session_state.update({"collections": getcollection(st.session_state.db_select)})
        )

        with st.form(key='RAG_form', clear_on_submit=True):
            st.write('RAG')
            #col1b, col2b = st.columns(2)

            # Obtener lista de colecciones para la base de datos seleccionada
            vector_collection = st.selectbox(
                'Select collection',
                options=st.session_state.collections,
                key='collection_select'
            )

            rag_question = st.text_area('Enter RAG question')
            rag_answer = st.text_input('RAG answer', disabled=False, on_change=None, value='')

            if st.form_submit_button('Create RAG'):
                st.write(f"RAG question: {rag_question}")
                st.write(f"RAG answer: {rag_answer}")
                # Procesamiento adicional si es necesario
                # mlv = DBConnection(vector_index, vector_collection)
                # mlv.process()

if __name__ == '__main__':
    main()
