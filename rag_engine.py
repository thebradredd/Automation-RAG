from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from ollama_embedding import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# Load shared objects once (faster)
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
model = OllamaLLM(model="gemma3")


def run_rag(query_text: str):
    """Execute full RAG: retrieve → build prompt → LLM → response"""

    # 1. Retrieve
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # 2. Build prompt
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 3. LLM
    answer = model.invoke(prompt)

    # 4. Sources
    sources = [doc.metadata.get("id") for doc, _ in results]

    return {
        "answer": answer,
        "sources": sources,
        "context": context_text,
        "prompt": prompt,
    }
