import streamlit as st
import os
import io
import fitz  # PyMuPDF
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load external CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# HTML templates for user and bot messages
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://static-00.iconduck.com/assets.00/user-avatar-robot-icon-2048x2048-ehqvhi4d.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUjQI66XbavSGAOfG7gKG-CvYNDj_BVgp5jg&s" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Function to save uploaded files to a temporary directory
def save_uploaded_file(uploaded_file):
    save_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Function to extract text and images from PDF with page numbers using PyMuPDF
def get_pdf_text_and_images(pdf_docs):
    text_with_page_numbers = []
    images_with_page_numbers = []
    
    for pdf in pdf_docs:
        pdf_path = save_uploaded_file(pdf)
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Extract text
            text = page.get_text()
            if text:
                text_with_page_numbers.append((text, page_num + 1))
            
            # Extract image
            img = page.get_pixmap()
            img_bytes = img.tobytes("png")  # Convert Pixmap to PNG bytes
            img_pil = Image.open(io.BytesIO(img_bytes))
            images_with_page_numbers.append((img_pil, page_num + 1))
    
    return text_with_page_numbers, images_with_page_numbers

# Function to split text into chunks
def get_text_chunks(text_with_page_numbers):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_page_numbers = []
    for text, page_num in text_with_page_numbers:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_page_numbers.append((chunk, page_num))
    return chunks_with_page_numbers

# Function to create a vector store
def get_vector_store(chunks_with_page_numbers):
    texts = [chunk for chunk, _ in chunks_with_page_numbers]
    page_numbers = [page_num for _, page_num in chunks_with_page_numbers]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=[{'page_num': num} for num in page_numbers])
    vector_store.save_local("faiss_index")
    return page_numbers

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context". Do not provide the wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and update chat history
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Build the context from chat history
    context = ""
    if "chat_history" in st.session_state:
        for chat in st.session_state.chat_history:
            context += f"Question: {chat['question']}\nAnswer: {chat['answer']}\n"

    # Add new documents context to the existing context
    context += "\n".join([doc.page_content for doc in docs])

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question, "context": context},
        return_only_outputs=True
    )

    answer_text = response["output_text"]
    relevant_page_numbers = [doc.metadata['page_num'] for doc in docs]

    # Check if the answer is not available in the context
    if "Answer is not available in the context" in answer_text:
        answer_text += "\nPlease check the images provided below for more information."

    # Use HTML templates for chat messages
    user_msg = user_template.replace("{{MSG}}", user_question)
    bot_msg = bot_template.replace("{{MSG}}", answer_text)

    st.markdown(user_msg, unsafe_allow_html=True)
    st.markdown(bot_msg, unsafe_allow_html=True)

    # Store question and response in chat history with timestamp
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "question": user_question, 
        "answer": answer_text, 
        "pages": relevant_page_numbers,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Display relevant images in an expander for the current response
    with st.expander("View Images", expanded=False):
        if "images_with_page_numbers" in st.session_state:
            for page_num in relevant_page_numbers:
                for img, img_page_num in st.session_state.images_with_page_numbers:
                    if img_page_num == page_num:
                        st.image(img, caption=f"Page {page_num}")

# Function to display chat history
def display_chat_history():
    if "chat_history" in st.session_state:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            timestamp = chat.get('timestamp', 'Timestamp not available')
            question = chat.get('question', 'No question available')
            answer = chat.get('answer', 'No answer available')
            pages = chat.get('pages', [])

            st.write(f"**Timestamp:** {timestamp}")
            user_msg = user_template.replace("{{MSG}}", question)
            bot_msg = bot_template.replace("{{MSG}}", answer)
            st.markdown(user_msg, unsafe_allow_html=True)
            st.markdown(bot_msg, unsafe_allow_html=True)
            
            # Create an expandable section for images
            with st.expander("View Images", expanded=False):
                if "images_with_page_numbers" in st.session_state:
                    for page_num in pages:
                        for img, img_page_num in st.session_state.images_with_page_numbers:
                            if img_page_num == page_num:
                                st.image(img, caption=f"Page {page_num}")

# Ensure chat_history is initialized at the start
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main function to create the Streamlit app
def main():
    st.set_page_config(page_title="EPINET Chat Bot", page_icon="🤖", layout="wide")

    # Load external CSS
    load_css()

    st.header("EPINET Chat Bot 🤖")
    st.markdown("#### Get answers to your queries regarding EPINET")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("📂 Upload PDF File")
        st.markdown("### Upload and Process")
        st.markdown("""
        **Welcome!** 

        Upload the PDF file shared in zip file here. Click on **'Submit & Process'** to start processing the uploaded file. 
        This will extract text and images from the PDF and prepare them for querying.
        """)
        
        pdf_docs = st.file_uploader("Choose PDF File", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing..."):
                    if not os.path.exists("temp_uploads"):
                        os.makedirs("temp_uploads")
                    text_with_page_numbers, images_with_page_numbers = get_pdf_text_and_images(pdf_docs)
                    chunks_with_page_numbers = get_text_chunks(text_with_page_numbers)
                    get_vector_store(chunks_with_page_numbers)
                    # Store images in session state
                    st.session_state.images_with_page_numbers = images_with_page_numbers
                    st.success("Processing complete!")
            else:
                st.warning("Please upload at least one PDF file before processing.")

    # User input section
    user_question = st.text_input("Ask a Question", placeholder="Type your question here...")

    if user_question:
        user_input(user_question)

    # Display chat history
    display_chat_history()

if __name__ == "__main__":
    main()
