import chromadb

class VectorStore:
    def __init__(self, collection_name="DarkPsychology_RAG"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents):
        """Generates IDs and populates the collection."""
        ids = [f"id_{i}" for i in range(len(documents))]
        self.collection.add(ids=ids, documents=documents)
        print(f"Successfully ingested {len(documents)} document chunks.")

    def has_documents(self):
        """Check if the collection already has data."""
        return self.collection.count() > 0

    def query(self, user_query, n_results=3):
        """Retrieves the most relevant context for a query as a flat list of strings."""
        results = self.collection.query(
            query_texts=[user_query],
            n_results=n_results
        )
        # Flatten the nested list returned by ChromaDB
        return results['documents'][0]