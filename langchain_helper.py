from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.llms import GooglePalm
vectordb_file_path = "faiss_index"
#initiating the llm model 
llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"], temperature=0.1)
embeddings = HuggingFaceInstructEmbeddings()
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)
    retriever = vectordb.as_retriever()
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt":PROMPT}
           )
    return chain

def create_vectordb():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column = 'prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding = embeddings)
    vectordb.save_local(vectordb_file_path)

if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain("do you have an EMI option?"))