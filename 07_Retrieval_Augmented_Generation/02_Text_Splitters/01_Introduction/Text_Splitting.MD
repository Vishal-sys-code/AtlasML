# **Text Splitting**
It is the process of breaking large chunks of text(like: articles, Pdfs, books etx) into smaller, manageable pieces (chunks) that an LLMs can handle effectively.

**Why it is used?**
- **Overcoming model limitations**: Many embeddings models and language models have max input size constraints, splitting allows us to process documents that would otherwise exceed these limit.
- **Downstream Tasks**: Text splitting improves nearly every LLM powered tasks
- **Optimizing Computational Resources**: Working with smaller chunks of text can be more memory efficient and allow for better parallelization of processing tasks.

**Types of Text Splitters**
- Length Based
- Text Structure Based
- Document Structure Based
- Semantic Meaning Based