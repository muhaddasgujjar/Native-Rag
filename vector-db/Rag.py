import chromadb
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the vector database client and target collection
client = chromadb.Client()
collection = client.get_or_create_collection(name="DarkPsychology_RAG") 

# Set up the LLM client (Note: Keep your API key secure in production!)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the research paper
pdf_path = "dark pyscology.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Configure the text splitter for dense academic reading
# 1600 characters ~ 400 tokens, 200 character overlap ensures no context is lost
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,      
    chunk_overlap=200,    
    separators=["\n\n", "\n", r"(?<=\. )", " ", ""] 
)

# Process the document into manageable chunks
chunked_docs = text_splitter.split_documents(pages)
documents = [chunk.page_content for chunk in chunked_docs]
ids = [f"id_{i}" for i in range(len(documents))]

print(f"Successfully created {len(documents)} document chunks.")

# Populate the ChromaDB collection
collection.add(ids=ids, documents=documents)

# Prompt the user for a query and search the vector database
user_query = input("\nAsk a question about the research paper: ")
results = collection.query(
    query_texts=[user_query], 
    n_results=1
)


prompt = f"""
CONTEXT: {results['documents']}
QUERY: {user_query}

Please generate a response based on the context and the query.Your response according to ROSE framework 
If the CONTEXT does not answer the query, politely say that you don't have the answer.
"""

# Generate the response using Groq and Llama 3.3
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n--- RESPONSE ---")
print(response.choices[0].message.content)