from extraction import extract_and_chunk
from ingestion import VectorStore
from generation import Generator
from dotenv import load_dotenv
import os

load_dotenv()
PDF_PATH = "dark pyscology.pdf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def main():
    # 1. Extraction
    documents = extract_and_chunk(PDF_PATH)
    # 2. Ingestion
    db = VectorStore()
    db.add_documents(documents)
    # 3. Retrieval
    user_query = input("\nAsk a question about the research paper: ")
    search_results = db.query(user_query, n_results=3)
    # Flatten the nested list from ChromaDB into a single string
    context = "\n\n".join(search_results['documents'][0])
    # 4. Generation
    bot = Generator(GROQ_API_KEY)
    answer = bot.get_response(context, user_query)
    
    print("\n--- RESPONSE ---")
    print(answer)

if __name__ == "__main__":
    main()