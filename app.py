import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import requests  # Import the requests library

# Define the RapidAPI endpoint and headers with your API key
rapidapi_endpoint = "https://gpt-text-generation.p.rapidapi.com/completions"
headers = {
    "X-RapidAPI-Key": "ENTER_API_KEY",
}

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [RapidAPI](https://www.rapidapi.com/) chatbot service  # Replace with the actual RapidAPI service URL

    ''')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf') 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Get store name from the PDF file name
        store_name = pdf.name[:-4] 
        embeddings_data= None 
        VectorStore= None

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            query_key = f"query_{store_name}"  # Create a unique key based on the PDF name 
            query = st.text_input("Ask questions about your PDF file ({query_key}):")

            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

                # Make a POST request to the RapidAPI service
                response = requests.post(rapidapi_endpoint, headers=headers, data={"question": query})

                if response.status_code == 200:
                    # Extract embeddings data from the response
                    embeddings_data = response.json()
                else:
                    st.write("Error: Failed to retrieve a response from the RapidAPI service")

            if embeddings_data is not None: 
                if VectorStore is not None:
                    # Create VectorStore using embeddings_data
                    VectorStore = create_vector_store(embeddings_data)

                    # Save the VectorStore to a file for future use
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(VectorStore, f)
        

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Make a POST request to the RapidAPI service
            response = requests.post(rapidapi_endpoint, headers=headers, data={"question": query})

            if response.status_code == 200:
                response_data = response.json()
                chatbot_response = response_data["answer"]
                st.write(chatbot_response)
            else:
                st.write("Error: Failed to retrieve a response from the RapidAPI service")

if __name__ == '__main__':
    main()
