# qna_review_apps
Extracting actionable insights from user reviews in apps

## How to Run the Chatbot
Prepare virtual environment by using:
```bash
conda create -n <environment-name> --file req.txt
```

To run locally, use:
```bash
streamlit run main.py
```

## Code Flow 
### Q&A Chatbot Project Tools:
- OpenAI LLM (for embedding and reasoning)
- RAG Tools: Langchain
- Vector DB Index: Pinecone

### Steps:
1. **Data Exploration and Preprocessing**
   - Located in the `exploration` folder
   - Get some insights and perform text preprocessing

2. **Embedding and Storage**
   - `ingestion.py`
   - Embed all the text reviews and store them in Pinecone

3. **Retrieval Stage**
   - Located in the `recall` folder
   - `core.py`
   - Create a retriever for querying the top 10 most similar reviews to the query
   - Create a prompt chain with user question input and retrieved context for the result

4. **UI**
   - `main.py`
   - For interactive question and answering process using Streamlit
