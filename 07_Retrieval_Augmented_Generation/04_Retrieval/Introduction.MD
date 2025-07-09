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

## **Maximum Marginal Relevance Retriever**
"How can we pick results that are not only relevant to the query but also different from each other"

MMR is an information retrieval algotithm designed to reduce redundancy in the retrieved results while maintaining high relevance to the query.

**Why MMR Retriever?**
In regular similarity search you may get documents that are:
- ALl very similar to each other
- Repeating the same info
- Lacking diverse perspectives

MMR Retriever avoids that by:
- Picking the most relevant documents first
- Then picking the nect most relevant and least similar to already selected docs.
- And so on...

This helps especially in RAG pipelines where:
- You want your context window to contain diverse but still relevant information
- Especially useful when documents are semantically overlapping.

## **Multi-Query Retriever**
"Sometimes a single query might not capture all the ways information is phrased in your documents"

**How does it works?**
- Takes your original query
- Uses an LLM (e.g. GPT 3.5) to generate multiple sementically different version of that query.
- Performs retrieval for each sub-query.
- Combines and de-duplicates the results.

## **Contextual Compression**

The Contextual Compression retriever in Langchain is an advanced retriever that improves retrieval quality by compressing documents after retrieval keeping only the relevant content based on the user's query.

**How it works**
- Base retrievers (e.g. FAISS, Chroma) retrieves N documents
- A compressor (usually an LLM) is applied to each deocument
- The compressor keeps only the parts relevant to the query.
- Irrelevant content is disabled.

**When to use**
- Your documents are long and contain mixed information.
- You want to reduce context length for LLMs.
- You need to improve answer accuracy in RAG pipelines.