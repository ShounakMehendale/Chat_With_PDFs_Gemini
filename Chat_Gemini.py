import streamlit as st
from pypdf import PdfReader
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableMap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv


def get_pdf_text(pdf_docs):
    text=""
    
    pdf_reader=PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=20)
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vector_database(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #vectordb=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    vectordb=FAISS.from_texts(text_chunks,embedding=embeddings)
    return vectordb

def get_rag_chain(retriever):
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know based on the context but you can provide the answer if you know the answer without this context.Use 8 sentences maximum No need to keep the answer precise"
    "\n\n"
    "{context}"
    )
    output=StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    rag_chain =RunnableMap(
        {
        "context": lambda x : retriever.similarity_search(x["input"],k=3),
        "input": lambda x : x["input"]

    }) | prompt | model | output
 
    return rag_chain

def handle_userinput(user_question):
    #final_answer=""
    
    #for i in range(9):
    #while(len(final_answer)<1000):
    response=st.session_state.rag.invoke({"input":user_question})
        #final_answer+=response['answer']
        #user_question=final_answer
    
    
    st.write(response)

def handle_without_pdf(user_question):
    llm=genai.GenerativeModel('gemini-1.5-pro')
    response=llm.generate_content(user_question)
    st.write(response.text)

def handle_image(query,image):
    model=genai.GenerativeModel("gemini-1.5-flash")
    response=model.generate_content([query,image])
    st.write(response.text)
    




def main():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    st.set_page_config(page_title="Chat with multiple PDFs/Images",page_icon=":books:")
    if "rag" not in st.session_state:
        st.session_state.rag=None
    st.header("Chat with your PDFs/Images :books:")
    user_question=st.text_input("Ask a question from these PDFs")
    
    
    
    pdf_docs = st.session_state.get("pdf_docs", [])
    with st.sidebar:
        st.subheader("Your Documents/Images")
        
        pdf_docs=st.file_uploader("Upload your files here")
        pdf_docs=pdf_docs
        if pdf_docs:
            name=pdf_docs.name
            
        if st.button("Upload"):
            #st.write(pdf_docs.type)
            with st.spinner("Processing"):
                if pdf_docs:
                    if  pdf_docs.name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff','.webp')):
                        st.write("Processed Successfully !")
                    elif pdf_docs.name.endswith('.pdf'):
                        
                            #get_pdf_text
                        raw_text=get_pdf_text(pdf_docs)
                        

                        #get_text_chunks
                        text_chunks=get_text_chunks(raw_text)
                        

                        #get_vector database
                        vectordb=get_vector_database(text_chunks)
                        st.write("Processed Successfully !")
                        retriever=vectordb

                        #get conversation chain
                        st.session_state.rag=get_rag_chain(retriever)

    if pdf_docs :
        
        name=pdf_docs.name
        
        if  name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff','.webp')) and  user_question:
            image=Image.open(pdf_docs)
            st.image(image,"Your Uploaded Image")
            handle_image(user_question,image)
        elif name.endswith not in ['png','jpeg','jpg'] and  user_question:
            handle_userinput(user_question)
    elif user_question:
        handle_without_pdf(user_question=user_question)   


if __name__=='__main__':
    main()