

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ml-papers-index-1")

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.8, groq_api_key=os.getenv("GROQ_API_KEY"))

embeddings_model = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

PAPER_MAPPING = {
    "Attention is all you need": "Attention is all you need",
    "BERT": "BERT",
    "CLIP": "CLIP",
    "GPT-3": "GPT-3",
    "LLaMA": "LLaMA"
}

def filter_guesser(query_input, paper_map):
    matched_ids = []
    for friendly_name, file_id in paper_map.items():
        if friendly_name.lower() in query_input.lower():
            matched_ids.append(file_id)

    if matched_ids:
        print(f"Found match. Filtering by paper ID(s): {matched_ids}")
        return matched_ids
    else:
        print("Couldn't find any specific paper. Will scan all.")
        return list(paper_map.values())

query_input = input("enter required query: ")
query_embedding = embeddings_model.embed_query(query_input)

target_paper_ids = filter_guesser(query_input, PAPER_MAPPING)

print(f"Querying Pinecone with filter: {{'paper_id': {{'$in': {target_paper_ids}}}}}")

query_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={"paper_id": {"$in": target_paper_ids}}
)

query_data = [match['metadata']['text'] for match in query_results['matches']]
data = " ".join(query_data)
def answer_question(query_input, context_text):

    template = """
    Answer in detail to the following question based ONLY on the context provided.The answer is required to be atleast of 50 words

    Context:
    {context}

    Question:
    {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    

    qa_chain = prompt | llm | StrOutputParser()

    final_answer = qa_chain.invoke({
        "context": context_text,
        "question": query_input
    })

    return final_answer

print("\n--- Generating Final Answer ---")
final_result = answer_question(query_input, data)
print("\n--- Answer ---")
print(final_result)