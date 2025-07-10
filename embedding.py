import os
import pinecone
from pinecone import Pinecone
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import time
import re

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ml-papers-index"
pdf_dir = "pdfs"  



def extract_my_content(pdf_path):
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True, x_tolerance=2)
                if page_text:
                    cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
                    cleaned_text = re.sub(r'-\s+', '', cleaned_text)
                    text += cleaned_text + '\n'
        return text
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")
        return ''

def chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ml-papers-index-1"


if index_name not in pc.list_indexes():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
    import time
    time.sleep(10)  # let the index initialize
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

index = pc.Index(index_name)



print("database created")

embeddings_model = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

print("cohere imported")
all_vectors = []
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, file)
        paper_id = os.path.splitext(file)[0]  

        text = extract_my_content(pdf_path)
        if not text:
            continue
        text_chunks = chunks(text)

        embeddings_list = []

        for i in range(0, len(text_chunks), 5):  # 5 chunks per batch
            chunk_batch = text_chunks[i:i+5]
    
        try:
            batch_embeddings = embeddings_model.embed_documents(chunk_batch)
            embeddings_list.extend(batch_embeddings)
        except Exception as e:
            print(f"Error embedding batch {i//5 + 1}: {e}")
        continue

        time.sleep(5)

        vectors = [
            {
                'id': f'{paper_id}-chunk-{i}',
                'values': emb,
                'metadata': {
                    'text': chunk,
                    'paper_id': paper_id  
                }
            }
            for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings_list))
        ]

        all_vectors.extend(vectors)
print("...................")
batch_size = 500
count=0
for i in range(0, len(all_vectors), batch_size):
    batch = all_vectors[i:i + batch_size]
    index.upsert(vectors=batch)
    print("batch 1 upserted")
    print(f"Upserted batch {count+1}")
    count+=1

time.sleep(10)
print("\n Index Stats:")
print(index.describe_index_stats())
