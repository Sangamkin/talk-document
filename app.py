import streamlit as st
import google.generativeai as genai
import time

from utils.document_loader import load_pdf, load_docx, load_txt, chunk_text
from utils.terxt_embedder import embed_chunks, retrieve_relevant_chunks

# Page config
st.set_page_config(page_title='Talk with your document', layout='wide')
st.title('Talk with your document')

# Sidebar for Gemini API key
st.sidebar.header('Gemini API Setup')
user_api_key = st.sidebar.text_input('Enter your Gemini API key', type='password')

if not user_api_key:
    st.warning('Please enter your Gemini key in the sidebar to start.')
    st.stop()

try:
    genai.configure(api_key=user_api_key)
    chat_model = genai.GenerativeModel('gemini-2.0-flash-lite')
    st.sidebar.success('Gemini API configured')
except Exception as e:
    st.sidebar.error(f'Error: setup failed: {e}')
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Check if new file is uploaded
if uploaded_file is not None:
    if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded_file.name:
        st.session_state.clear()  # Clear previous state for new document
        st.session_state['last_uploaded'] = uploaded_file.name

        # Load file content
        if uploaded_file.name.lower().endswith(".pdf"):
            text = load_pdf(uploaded_file)
        elif uploaded_file.name.lower().endswith(".docx"):
            text = load_docx(uploaded_file)
        elif uploaded_file.name.lower().endswith(".txt"):
            text = load_txt(uploaded_file)
        else:
            st.error('Unsupported file type.')
            st.stop()

        # Chunk and embed
        st.info('Chunking and embedding document...')
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        st.session_state['chunks'] = chunks
        st.session_state['embedding'] = embeddings
        st.session_state['document_processed'] = True

        st.success(f"File '{uploaded_file.name}' uploaded and processed.")
        # st.text_area("Document Preview", text[:2000], height=200)

# Ask questions
if st.session_state.get('document_processed', False):
    st.subheader('Ask a question about your document')
    query_key = f"query_input_{st.session_state['last_uploaded']}"
    query = st.text_input('Enter your question:', key=query_key)

    if query:
        top_chunks = retrieve_relevant_chunks(
            query,
            st.session_state['chunks'],
            st.session_state['embedding'],

        )
        context = "\n\n".join(top_chunks)
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"

        st.info('Generating answer with Gemini...')
        st.markdown('### Answer')
        response_area = st.empty()
        try:
            response_stream = chat_model.generate_content(prompt, stream=True)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_area.markdown(full_response)
                    time.sleep(0.05)
        except Exception as e:
            st.error(f"Error generating response: {e}")
else:
    st.info('Please upload and process a document to start asking questions.')
