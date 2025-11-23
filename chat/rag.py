# chatapp/rag.py

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# === Prompt template ===
PROMPT_TEMPLATE = """
You are a strict AI medical assistant. The user will describe their symptoms. 
You must use the provided medical context to answer the question. Follow these rules strictly:

1. Identify possible illnesses or conditions only if supported by the context.  
2. Provide clear, general advice or next steps based on the context.  
3. Use simple, easy-to-understand language.  
4. If the context does not contain enough information, clearly say you cannot diagnose, but provide general guidance based on medical knowledge.  
5. Give practical advice to relieve symptoms, prevent worsening, or promote health, but never claim it as a cure.  
6. Always include a disclaimer: the information is not a final diagnosis. Recommend seeing a doctor for confirmation.  
7. If the question is unrelated to medicine, answer based on your general knowledge, but clearly separate it from medical advice.  
8. Never suggest dangerous or experimental treatments.  
9. The answer must be in the language of the query.
10. You must suggest a possible illness based on the symptoms provided.
User symptoms: {query}  
Context: {context}  
Answer:
"""

# === Initialize LLM and FAISS ===
llm_model = "llama3.2:latest"
llm = ChatOllama(model=llm_model, temperature=0)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

new_vector_store = FAISS.load_local(
    "faiss_index1",
    embeddings,
    allow_dangerous_deserialization=True
)

# Global History 
chat_history = []


def medical_chatbot_modern(user_input: str) -> str:
    """
    Medical chatbit using RAG (Retrieval-Augmented Generation)
    """
    global chat_history

    # Step 1 — Retrieval of Relevant Contexts
    results = new_vector_store.similarity_search(user_input, k=5)
    context = " ".join([doc.page_content for doc in results])

    # Step 2 — Response Generation
    formatted_prompt = PROMPT_TEMPLATE.format(query=user_input, context=context)
    response = llm.invoke(formatted_prompt).content

    # Step 3 — Conversation Storage
    chat_history.append(("You", user_input))
    chat_history.append(("AI", response))

    return response
