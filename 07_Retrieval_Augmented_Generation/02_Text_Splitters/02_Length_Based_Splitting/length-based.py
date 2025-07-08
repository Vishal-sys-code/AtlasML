from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pypdf
loader = PyPDFLoader('book.pdf')

docs = loader.load()

# RecursiveCharacterTextSplitter is recommended for general text.
# It attempts to split on semantic boundaries (paragraphs, sentences, words).
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

result = splitter.split_documents(docs)
print(f"The PDF was split into {len(result)} documents.")

if result:
    print("\n--- First chunk ---")
    print(result[0].page_content)