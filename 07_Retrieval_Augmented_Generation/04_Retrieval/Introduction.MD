# **Introduction**
A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user's query.

There are multiple types of retrievers.

All retrievers in Langchain are runnables

## **Types of Retrievers**
- **On the basis of Data Source:**
    - Wikipedia Retriever
    - Vector Store Based Retriever
    - Arxiv Retriever
- **On the basis of Search Strategy:**
    - Maximum Marginal Relevance Retriever
    - Contextual Compression
    - Multi-Query Retriever

***Langchain has more than 25 retrievers. Use them wisely as per your requirements...***

## **Wikipedia Retriever**
A wikipedia retriever is a retriever that queries the wikipedia API to fetch relevant content for a given query.
**How it works**
- You give it a query (e.g: 'Albert Einstein')
- It sends the query to Wikipedia's API
- It retrieves the most relevant articles
- It returns them as Langchain Document objects

## **Vector Store Retriever**
A vector store retriever in langchain is the most common type of retriever that lets you search and fetch documents from a vector store based on semantic similarity using vector embeddings
**How it works**
- You store your documents in a vector store (like: FAISS, Chroma, Weaviate)
- Each document is converted into a dense vector using an embedding model
- When the user entries a query:
    - It's also turned into a vector
    - The retriever compares the query vector with the stored vectors
    - It retrives the top-k most similar ones