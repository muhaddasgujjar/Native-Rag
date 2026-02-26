from groq import Groq

class Generator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def get_response(self, context, query):
        """Generates a response using the ROSE framework.
        
        Args:
            context: A string containing relevant document chunks.
            query: The user's question.
        """
        prompt = f"""You are a helpful research assistant. Answer the user's question using ONLY the provided context.

CONTEXT:
{context}

QUESTION: {query}

Use the ROSE framework for your answer:
- R (Role): You are an expert analyst of the provided document.
- O (Objective): Answer the question accurately based on the context.
- S (Specifics): Use direct references and details from the context. Be concise.
- E (Examples): Include relevant quotes or examples from the context when helpful.

If the context does not contain enough information to answer the question, say so clearly.
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You answer questions based strictly on the provided context. Do not make up information."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
