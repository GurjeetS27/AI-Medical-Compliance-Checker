import streamlit as st
import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import time
import fitz

# Load API Keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

# Initialize OpenAI API
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # ✅ Updated for latest OpenAI API

# Initialize Pinecone
pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("compliance-regulations")  # Ensure this matches your Pinecone index name

# ✅ Extract text from uploaded files (PDF or TXT)
def extract_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf_doc:
            text += page.get_text()
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    return text.strip()

# ✅ Convert Text to Embeddings using OpenAI (Latest Version)
def create_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# You can manually store compliance documents here example
# COMPLIANCE_DOCUMENTS = [
#     {"id": "HIPAA-001", "text": "Under HIPAA, patients have the right to access their medical records within 30 days of a request."},
# ]

# ✅ Store Compliance Documents in Pinecone
# def store_compliance_documents():
#     for doc in COMPLIANCE_DOCUMENTS:
#         embedding = create_embedding(doc["text"])
#         if embedding:
#             index.upsert(vectors=[{"id": doc["id"], "values": embedding, "metadata": {"text": doc["text"]}}])
#             print(f"✅ Stored document {doc['id']} in Pinecone.")

# ✅ Store Extracted Text in Pinecone
def store_uploaded_document(file, user_key):
    """Handles document uploads: Admins can store in Pinecone, users get temporary usage."""
    text = extract_text_from_file(file)
    if not text.strip():
        return "⚠️ The uploaded document is empty or non-readable. Please upload a valid compliance document."
    embedding = create_embedding(text)
    if not embedding:
        return "⚠️ Failed to process document embedding."
    doc_id = file.name
    # ✅ Admins can permanently store documents in Pinecone
    if user_key == ADMIN_SECRET:
        index.upsert(vectors=[{"id": doc_id, "values": embedding, "metadata": {"text": text}}])
        return f"✅ Successfully stored in Pinecone: {doc_id}"
    # ✅ Regular users: Use document only for this session (do not store in Pinecone)
    return f"⚠️ Temporary Use Only: {doc_id} will not be stored in Pinecone."

# ✅ Search for Compliance Documents in Pinecone
def search_compliance_documents(query):
    """Searches Pinecone only if the index contains compliance documents."""
    if is_pinecone_empty():  # Check if Pinecone has any stored data
        print("\n⚠️ Pinecone is empty. Skipping search and defaulting to OpenAI.")
        return None  
    query_embedding = create_embedding(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    print("\n🔍 Pinecone Search Results:", search_results) 
    if search_results and search_results["matches"]:
        best_match = search_results["matches"][0]
        similarity_threshold = 0.45  
        if best_match["score"] >= similarity_threshold:
            # print(f"\n✅ Using Pinecone Result (Score: {best_match['score']}): {best_match['metadata']['text']}")
            return best_match["metadata"]["text"]
    # print("\n❌ No relevant result found in Pinecone, using OpenAI only.")
    return None  


def is_compliance_related(question):
    """Quickly checks if the question is about compliance before using OpenAI."""
    compliance_keywords = ["HIPAA", "compliance", "policy", "law", "regulation", "GDPR", "legal", "FDA", "data security", "privacy", "healthcare"]
    # ✅ Fast keyword check first
    if any(keyword.lower() in question.lower() for keyword in compliance_keywords):
        return True  # ✅ Instantly approve if keywords exist
    # ✅ If no keywords found, ask OpenAI
    system_prompt = "You are a classifier that determines if a question is related to compliance, legal policies, or regulations. Respond with 'YES' if it is compliance-related and 'NO' if it is not."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        classification = response.choices[0].message.content.strip().upper()
        return classification == "YES"
    except Exception as e:
        print(f"⚠️ OpenAI Classification Error: {e}")
        return False  # If classification fails, assume it's not compliance-related


# ✅ Get Compliance Answer (Uses Pinecone + OpenAI)
def is_pinecone_empty():
    """Checks if the Pinecone index has any stored documents."""
    try:
        index_stats = index.describe_index_stats()
        total_vectors = index_stats.get("total_vector_count", 0)  # ✅ Uses .get() to avoid KeyErrors
        return total_vectors == 0  # ✅ Returns True if Pinecone is empty
    except Exception as e:
        print(f"⚠️ Error checking Pinecone index: {e}")
        return True  # Assume empty if there’s an error

def get_compliance_answer(question):
    """Generates compliance-related answers only after verifying Pinecone has data."""
    
    # ✅ Step 1: Check if Pinecone is empty
    if is_pinecone_empty():
        return "⚠️ No compliance documents uploaded. Please upload a compliance document before asking questions.\n\n(Source: No Data 🚫)"

    # ✅ Step 2: Use AI to classify if the question is compliance-related
    if not is_compliance_related(question):
        return "⚠️ This AI is only for compliance-related questions. Please ask about compliance policies, regulations, or legal matters.\n\n(Source: Restricted 🚫)"

    # ✅ Step 3: Search for compliance document in Pinecone
    relevant_document = search_compliance_documents(question)

    # ✅ NEW: Optimized OpenAI call to format response directly
    prompt = f"""
    Provide a structured and well-formatted compliance response.
    Use bullet points, section headers, and a final summary for clarity.
    Ensure legal terminology is precise.

    Compliance Regulation: {relevant_document if relevant_document else 'No relevant document found'}.

    Question: {question}
    """
    source = "Pinecone ✅" if relevant_document else "OpenAI Only ❌"

    attempts = 0
    while attempts < 2:  # ✅ NEW: Retry mechanism for OpenAI failures
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return f"{response.choices[0].message.content}\n\n(Source: {source})"
        except openai.error.OpenAIError as e:
            print(f"⚠️ OpenAI API Error: {e}")
            time.sleep(2)  # Wait before retrying
            attempts += 1

    return "⚠️ OpenAI is currently unavailable. Please try again later.\n\n(Source: API Error 🚫)"


# ✅ Streamlit UI
st.set_page_config(page_title="AI Medical Compliance Checker", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
        /* Global Background */
        .reportview-container {
            background: #ffffff;  /* Clean white background */
        }

        /* Move the UI Up Slightly */
        .block-container {
            padding-top: 0rem !important;
            margin-top: 30px !important;
        }

        /* Text Area & Input Fields */
        .stTextArea textarea, .stTextInput input {
            font-size: 16px !important;
            padding: 12px !important;
            border: 1px solid #ccc !important;
            border-radius: 8px !important;
            box-shadow: none !important;  /* Remove extra shadows */
            transition: 0.2s ease-in-out;
            outline: none !important;  /* Removes extra blue outline */
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            box-shadow: 0px 0px 8px rgba(0, 123, 255, 0.3) !important;
            outline: none !important;  /* Ensures clean focus */
        }

        /* Button Styling */
        .stButton>button {
            font-size: 18px !important;
            background-color: #007BFF !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: bold !important;
            border: none !important;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #0056b3 !important;
            box-shadow: 0px 4px 12px rgba(0, 91, 187, 0.3) !important;
        }

        /* File Uploader */
        .stFileUploader {
            border: 2px solid #ddd !important;
            border-radius: 8px !important;
            padding: 10px !important;
            background-color: #f8f9fa !important;
            transition: 0.3s ease-in-out;
        }
        .stFileUploader:hover {
            border: 2px solid #007BFF !important;
            background-color: #f0f8ff !important;
        }

        /* Admin Hint Text */
        .admin-hint {
            font-size: 12px !important;
            color: #777 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 AI Medical Compliance Checker")
st.write("### Upload compliance documents (PDF or TXT) to train the AI.")

# ✅ File Upload Section
uploaded_file = st.file_uploader("Upload Compliance Document (PDF or TXT)", type=["pdf", "txt"])
user_key = st.text_input("Enter Admin Key (Only for permanent storage)", type="password") # Hidden password field
st.markdown('<p class="admin-hint">Leave blank if you are not an admin.</p>', unsafe_allow_html=True)


if uploaded_file:
    result = store_uploaded_document(uploaded_file, user_key)
    st.success(result)

st.write("### Ask a compliance-related question:")

query = st.text_area("Enter your question here:", height=100)

if st.button("Check Compliance"):
    if query.strip():
        with st.spinner("🔍 Retrieving answer..."): #Loading indicator
            answer = get_compliance_answer(query)
        st.subheader("📝 Compliance Answer:")
        st.success(answer) #Displays answer in a highlightes box
    else:
        st.warning("⚠️ Please enter a question.")

# ✅ Run Once: Store compliance documents in Pinecone (Uncomment to run manually)
# store_compliance_documents()
