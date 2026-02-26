import chromadb

# 1. Initialize Client and Collection
client = chromadb.Client()
collection = client.get_or_create_collection(name="Feb-24")

# 2. Define Data
documents = [
    "Hi I am Muhammad Muhaddas and I am AI and ML Intern with over 4 months of experience",
    "I am proud to have gained over 4 months of hands-on experience in developing and testing machine learning models",
    "I am a student at Lahore Garrison University, where I am sharpening my technical and analytical foundations.",
    "I am passionate about leveraging data to solve real-world problems and am currently exploring advanced topics in artificial intelligence and deep learning.",
    "I am a quick learner and a team player, and I am always looking for new opportunities to grow and develop my skills.",
    "I am actively building a high-performance vector database project to explore advanced data retrieval.",
    "I am specializing in the use of ChromaDB to manage embeddings and improve search accuracy in AI applications.",
    "I am committed to staying at the forefront of the rapidly evolving tech landscape in Lahore.",
    "My goal is to contribute to innovative AI solutions that make a meaningful impact.",
    "I am excited about the future of AI and am eager to be part of the next wave of technological advancements."
]

ids = [f"id{i}" for i in range(len(documents))]

# 3. PRINT IDS AND CHUNKS FIRST
print("----> Database Content (Chunks) <----")
for i in range(len(documents)):
    print(f"ID: {ids[i]} | {documents[i]}")
print("-" * 50)

# 4. Add to Collection
collection.add(ids=ids, documents=documents)

# 5. USER INPUT (Happens after printing chunks)
user_query = input("\nAsk a question about Muhammad Muhaddas: ")

# 6. Query Process
results = collection.query(
    query_texts=[user_query], 
    n_results=1,
    include=['embeddings', 'documents', 'distances']
)

# 7. Display Results
print("\n" + "="*30)
print(f"QUERY: {user_query}")
print("="*30)

print("\n----> Top Matches Found <----")
for i, doc in enumerate(results['documents'][0]):
    distance = results['distances'][0][i]
    print(f"Match {i+1} (Score: {distance:.4f}):\n   {doc}\n")

print("----> Vector Data Sample <----")
print(f"Vector size: {len(results['embeddings'][0][0])}")
print(f"First 10 values: {results['embeddings'][0][0][:10]}")