"""
Este módulo define constantes para las rutas de directorios utilizadas en la aplicación.
"""

RAG_DIR = "rag_documents"
TEMP_DIR = "temp"
CHROMA_DB = "chroma_db"
CHROMA_COLLECTION = "documents"
STRUCTURE_DIRECTORY = '00_simulación_rag'
STRUCTURE_FILE = RAG_DIR + '/' + 'structure.txt'
SYSTEM_PROMPT_PATH = 'system_prompts_V2.json'
GOLD_STANDARD_PATH = 'gold_standard.json'

ALLOWED_EXTENSIONS = ['pdf', 'txt', 'json', '.pdf', '.json',
                      '.txt', 'csv', '.csv', 'xlsx', '.xlsx', 'xls', '.xls']
