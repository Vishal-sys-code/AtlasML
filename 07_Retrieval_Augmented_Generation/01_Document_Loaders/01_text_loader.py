import openai
from langchain_community.document_loaders import TextLoader 
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('../../00_Datasets/cricket.txt', encoding = 'utf-8')
docs = loader.load()

model = ChatOpenAI()
prompt =  PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)
parser = StrOutputParser()
chain = prompt | model | parser

# Invoke the chain with the document's content and store the result
summary = chain.invoke({'poem': docs[0].page_content})

# Print the generated summary
print("\n--- Generated Summary ---\n")
print(summary)