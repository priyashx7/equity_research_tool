import os
import streamlit as st
import pickle
import time
import langchain
from huggingface_hub import InferenceClient  # ✅ Correct import
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  
from langchain.llms.base import LLM
from typing import Optional, List, Any


from dotenv import load_dotenv
load_dotenv() # take environment varaible from .env

#Since we are not using the appi key of hugging face but still included,
# as probably if we go for increase in rate limit and at that time we may require to use api key for the paid version.

#set the page elements
st.title("Equity Research Tool 🚀")
st.sidebar.title("News Article URLs")


# ✅ Initialize session state for dynamic URL input
if "num_urls" not in st.session_state:
    st.session_state.num_urls = 3  # Start with 3 input fields

urls = []
for i in range(st.session_state.num_urls):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

# ✅ Button to add more URL fields
if st.sidebar.button("➕ Add More URLs"):
    st.session_state.num_urls += 1

process_url_clicked = st.sidebar.button("Process URLs")


main_placeholder = st.empty()

# ✅ Initialize the client (no parameters here)
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", 
    token=os.getenv("huggingface_api_key")  # ✅ Uses API key from .env
)

# ✅ Create a Custom LLM Wrapper
class HuggingFaceInferenceLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return client.text_generation(prompt, temperature=0.9)

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface_inference"

file_path= "faiss_store.pkl"

if process_url_clicked:

    #load the data
    loader =UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading....Started...✅✅✅")
    data = loader.load()

    #split the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )

    main_placeholder.text("Text Splitter....Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    #create embeddings and save it to faiss index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index_huggingface = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


    #save the faiss index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vector_index_huggingface, f)  # Use the correct FAISS index

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
            # ✅ Use the Custom LLM in Retrieval Chain
            llm = HuggingFaceInferenceLLM()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
            result = chain({"question":query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])


            #Display sources, if available
            sources = result.get("sources","")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)

