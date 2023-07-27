import sys

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

loader = PyPDFLoader("Profile.pdf")
docs = loader.load_and_split()

prompt_template = """

Please answer your questions about Stephen Jackewicz as sassily as possible, given the following context:

{context}

Question: {question}
Sassy Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
chain_type_kwargs

text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "What is the area of expertise for Stephen Jackewicz"
query = sys.argv[1]
docs = db.similarity_search(query)

embeddings = OpenAIEmbeddings()

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)

response = qa.run(query)
print(response)
