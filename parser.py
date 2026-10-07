from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def PDFparser(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    documents = []
    for page_number, page in enumerate(pages):
        documents.append({
            "page_number": page_number + 1,
            "content": page.page_content
        })

    return documents


