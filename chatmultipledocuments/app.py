import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_experimental.chat_models import Llama2Chat
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import openai
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))







def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text,model):
    if model=="Gemini":
        print('Chunking for Gemini')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    if model=="OpenAI":
         print('chunking for OpenAI')
         text_splitter = CharacterTextSplitter(
                separator = "\n",
                chunk_size = 800,
                chunk_overlap  = 200,
                length_function = len,
         )
         chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks,model):
    if model=="Gemini":
        print('Create embeddings for Gemini')
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    if model=="OpenAI":
        print('Create embeddings for OpenAI')
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_openai")


def get_conversational_chain(selectedmodel):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    if selectedmodel=="Gemini":
        model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3
                             )
    if selectedmodel=="OpenAI":
        model= openai.OpenAI(model_name="gpt-3.5-turbo-instruct")
        model_name = "unknown"
        if isinstance(model, openai.OpenAI):
            model_name = model.model_name
        print(f'Load embeddings for OpenAI using model: {model_name}')
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question,model):
    if model=='Gemini':
        print('Load embeddings for Gemini')
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    if model=='OpenAI':
        print('Load embeddings for OpenAI')
        embeddings= OpenAIEmbeddings()
        new_db = FAISS.load_local("faiss_index_openai", embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain(model)
    docs = new_db.similarity_search(user_question)
    response = chain(
    {"input_documents":docs, "question": user_question}
    , return_only_outputs=True)
    response=response["output_text"]
    
    print(response)
    st.write("Reply: ", response)




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLM💁")
    model= st.radio("Select a Model",["Gemini", "OpenAI"])
    print(model)

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question,model)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text,model)
                get_vector_store(text_chunks,model)
                st.success("Done")



if __name__ == "__main__":
    main()