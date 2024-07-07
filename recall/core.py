from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from constant import INDEX_NAME

def retrieval_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model='gpt-3.5-turbo-0125' ,verbose=True, temperature=0)

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
    [("system", "Your name is Spotify Apps Review Chatbot. Summarize and help answer questions based solely on the context below. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.:\n\n<context>\n{context}\n</context>\n\n")])

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    print(retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(search_kwargs={'k':10}), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = retrieval_llm(query="what is people opinion about the UI?")
    print(res)
