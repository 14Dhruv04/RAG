from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, CSVLoader

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


def Excelparser(file_path):
    loader = UnstructuredExcelLoader(file_path)
    pages = loader.load()

    documents = []
    for page_number, page in enumerate(pages):
        documents.append({
            "page_number": page_number + 1,
            "content": page.page_content
        })
        
    return documents

def CSVparser(file_path):
    loader = CSVLoader(file_path, encoding="utf-8")
    pages = loader.load()

    documents = []
    for page_number, page in enumerate(pages):
        documents.append({
            "page_number": page_number + 1,
            "content" : page.page_content
        })

    return documents