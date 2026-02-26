from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure this name is exactly 'extract_and_chunk'
def extract_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,      
        chunk_overlap=200,    
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""] 
    )

    chunked_docs = text_splitter.split_documents(pages)
    return [chunk.page_content for chunk in chunked_docs]