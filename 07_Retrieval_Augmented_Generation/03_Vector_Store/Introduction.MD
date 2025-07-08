## **Introduction**
A vector store is a system designed to store and retrieve data represented as numerical vectors.

**Key Features**
- **Storage:** Ensures that vectors and their associated metadata are retained, whether in-memory for quick lookups or on-disk for durability and large-scale use.
- **Similarity Search:** Helps retrieve the vectors most similar to a query vector
- **Indexing:** Provide a data-structure or method that enables fast similarity searches on high-dimensional vectors (e.g. approximate nearest neighbour llokups)
- **CRUD Operations:** Manage the lifecycle of data, adding new vectors, reading them, updating existing entries, removign outdated vectors

**USE-CASES**
- Semantic Search
- RAG
- Recommender Systems
- Images / Multimedia Search

<hr>

## **Vector Store vs Vector Database**

* **Vector Store**
    * Typically refers to a lightweight library or service that focuses on storing vectors(embeddings) and performing similarity search
    * May not include many traditional DB features like transactions, rich query languages or role-based access control.
    * Ideal for prototyping, similar-scale applications
    * Examples: FAISS(where you can store vectors and can query them by similarity, but you handle persistence and scaling seperately).

Remember => Vector Store gives you: Storage and Retrieval...
If you add some features => Distributed, backup and restore, ACID transaction, concurrency, authentication. => This will give the feature of the vector database.
[Vector Stores + Features of DB = Vector DB]

* **Vector Database**
    * A full fledged DB system designed to store and query vectors.
    * Offers additional "database-like" features:
        * Distributed architecture for horizontal scaling
        * Durability and persistence (replication, backup/restore)
        * Metadata handling (schemas, filters)
        * Potential for ACID or near-ACID guarantees
        * Authentication / Authorization and more advanced security
    * Geared for production environments with significant scaling, large datasets.
    * Examples: Pinecone, Qdrant, Milvus, Weaviate

A vector DB is effiectively a vector store with extra DB features(e.g. clustering, scaling, security, metadata filtering, and durability)