import nest_asyncio
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv, find_dotenv

ARCHIVO_PDF = "./PK/CURRENT NEURO PARKINSON.pdf"
ARCHIVO_SALIDA = "./PK/CURRENT NEURO PARKINSON.txt"


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
documents = parser.load_data(ARCHIVO_PDF)

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

filename = ARCHIVO_SALIDA
if documents:  # Verifica si la lista 'documents' no está vacía
    with open(filename, "w") as f:
        f.write(documents[0].text)
else:
    print("No se encontraron documentos.")