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
from PIL import Image
import io
import base64
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Function to configure API key
def configure_api_key(api_key):
    try:
        if not api_key:
            st.error("Please enter a valid Google Gemini API key.")
            return False
        genai.configure(api_key=api_key)
        # Test the API key by making a simple request
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Test API key functionality.")
        if response.text:
            st.success("API key validated successfully!")
            return True
        else:
            st.error("Invalid API key. Please check and try again.")
            return False
    except Exception as e:
        st.error(f"Error validating API key: {str(e)}")
        logger.error(f"API key validation error: {str(e)}")
        return False

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    if not pdf_docs:
        return ""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        logger.error(f"PDF text extraction error: {str(e)}")
        return ""

# Function to extract text from images using Gemini 1.5 Flash
def get_image_text(image_file):
    if not image_file:
        return None
    try:
        img = Image.open(image_file)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Analyze the text in the provided image. Extract all readable content and present it in a structured Markdown format (e.g., headings, lists, or code blocks) that is clear, concise, and well-organized.",
            {"mime_type": "image/png", "data": img_base64}
        ])

        if response.parts and len(response.parts) > 0:
            return response.text
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            st.warning(f"No text extracted from image. Finish reason: {finish_reason}")
            return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        logger.error(f"Image processing error: {str(e)}")
        return None

# Function to split text into chunks
def get_text_chunks(text):
    if not text:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {str(e)}")
        logger.error(f"Text splitting error: {str(e)}")
        return []

# Function to create and save vector store
def get_vector_store(text_chunks, ocr_text=None):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        all_texts = text_chunks
        if ocr_text:
            all_texts.append(ocr_text)
        if not all_texts:
            st.warning("No text available to create vector store.")
            return False
        vector_store = FAISS.from_texts(all_texts, embeddings)
        vector_store.save_local("faiss_index")
        logger.debug("Vector store created and saved.")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        logger.error(f"Vector store creation error: {str(e)}")
        return False

# Function to set up conversational chain
def get_conversation_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, ensuring all relevant details are included. 
        If the answer is not in the provided context, attempt to provide a general answer based on common knowledge, but clearly state that the information is not directly from the context. If no relevant information can be provided, state, "answer is not available in the context," and avoid providing incorrect information.

        You are an AI assistant that extracts and structures product dimension data from technical drawings, including part diagrams and engineering schematics. However, if the user asks unrelated questions (e.g., about recipes or general knowledge), provide a helpful response based on general knowledge if possible.

        Context: {context}
        Question: {question}
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error setting up conversation chain: {str(e)}")
        logger.error(f"Conversation chain setup error: {str(e)}")
        return None

# Function to handle user questions and return answers
def process_user_input(user_question):
    try:
        if not os.path.exists("faiss_index"):
            return "No data has been processed yet. Please upload and process files first."
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logger.debug("Loading FAISS index...")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logger.debug(f"Searching for documents with question: {user_question}")
        docs = new_db.similarity_search(user_question)
        logger.debug(f"Found {len(docs)} documents.")
        
        chain = get_conversation_chain()
        if chain is None:
            return "Error: Unable to set up the conversation chain."
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        logger.debug(f"Response generated: {response['output_text']}")
        return response["output_text"]
    except Exception as e:
        logger.error(f"Error in process_user_input: {str(e)}")
        return f"Error processing question: {str(e)}"

# Main application
def main():
    st.set_page_config(
        page_title="Chatbot for PDF and Images",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for user-friendly styling
    st.markdown("""
        <style>
        .stTextInput > div > input {
            border-radius: 5px;
            padding: 10px;
        }
        .stButton > button {
            border-radius: 5px;
            padding: 10px 20px;
        }
        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("Chatbot for PDF and Images using Gemini")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {"pdfs": [], "image": None}
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

    # API Key Input
    st.subheader("Configure API Key")
    with st.form(key="api_key_form"):
        api_key = st.text_input("Enter your Google Gemini API Key:", type="password", help="Your API key will be used to access the Gemini API.")
        submit_api_key = st.form_submit_button("Validate API Key")
        if submit_api_key and api_key:
            if configure_api_key(api_key):
                st.session_state.api_key_valid = True
            else:
                st.session_state.api_key_valid = False

    # Proceed only if API key is valid
    if not st.session_state.api_key_valid:
        st.warning("Please validate your API key to proceed.")
        return

    # Clear button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear 🗑️", help="Reset all data and start fresh"):
            st.session_state.chat_history = []
            st.session_state.uploaded_files = {"pdfs": [], "image": None}
            st.session_state.processed = False
            if 'ocr_result' in st.session_state:
                del st.session_state['ocr_result']
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
                logger.debug("FAISS index cleared.")
            st.rerun()

    # Sidebar for file uploads
    with st.sidebar:
        st.title("Menu")
        st.markdown("### Upload Files")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, key="pdf_uploader", help="Upload PDF files to extract text.")
        image_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'], key="image_uploader", help="Upload an image to extract text via OCR.")

        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.uploaded_files["image"] = image_file

        if pdf_docs:
            st.session_state.uploaded_files["pdfs"] = pdf_docs

        if st.button("Submit & Process", help="Process the uploaded files to extract text and prepare for chat."):
            if not pdf_docs and not image_file:
                st.error("Please upload at least one PDF or image to process.")
            else:
                with st.spinner("Processing files..."):
                    text_chunks = []
                    ocr_text = None
                    if st.session_state.uploaded_files["pdfs"]:
                        raw_text = get_pdf_text(st.session_state.uploaded_files["pdfs"])
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                        else:
                            st.warning("No text extracted from PDF.")
                    if st.session_state.uploaded_files["image"]:
                        ocr_text = get_image_text(st.session_state.uploaded_files["image"])
                        if ocr_text:
                            st.session_state['ocr_result'] = ocr_text
                    if text_chunks or ocr_text:
                        if get_vector_store(text_chunks, ocr_text):
                            st.session_state.processed = True
                            st.success("Files processed successfully!")
                        else:
                            st.error("Failed to process files. Please try again.")
                    else:
                        st.error("No content processed. Please check your files.")

    # Display extracted image content
    if 'ocr_result' in st.session_state:
        st.markdown("### Extracted Image Content")
        st.markdown(st.session_state['ocr_result'])

    # Chat interface
    st.markdown("### Chat with the Bot")

    # Display chat history with styled messages
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'>👤 <b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'>🤖 <b>Bot:</b> {message['content']}</div>", unsafe_allow_html=True)

    # User input for new question
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Type your question here:", key="user_input", help="Ask a question based on the uploaded files or any general topic.")
        submit_button = st.form_submit_button(label="Send")

        if submit_button and user_question:
            if not st.session_state.processed and "answer is not available in the context" not in user_question.lower():
                st.warning("Please process files before asking questions, unless seeking general knowledge.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                response = process_user_input(user_question)
                st.session_state.chat_history.append({"role": "bot", "content": response})
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("Powered by Google Gemini | Developed for Customer Deployment")

if __name__ == "__main__":
    main()