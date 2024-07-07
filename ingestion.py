from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from constant import INDEX_NAME
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

def ingest_docs():
    
    raw_documents = pd.read_csv('data/SPOTIFY_REVIEWS_CLEAN.csv')
    print(f"loaded {len(raw_documents)} reviews")
    
    texts = raw_documents['review_text'].to_list()

    # limitation is in the index storage, free tier only contains around 2gb data - 2 mio write units
    # writing process quite fast, overall ingestion probably took 6 hours
    PineconeVectorStore.from_texts(
        texts, embeddings, index_name=INDEX_NAME
    )

    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()

