# ✅ Disable Streamlit when running in test mode
import os
import logging
import mimetypes
import requests
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
import fitz
import time

# Set up the logger
logging.basicConfig(
    filename="compliance_checker.log",  # Log file
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

session = requests.Session() # Persistent session to reduce overhead

# Load API Keys from .env
load_dotenv()


# ✅ Detect if running in test mode
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

if TEST_MODE:
    print("✅ TEST MODE ENABLED: Streamlit UI components are disabled.")
else:
    import streamlit as st

    # ✅ Move all Streamlit UI setup inside this block
    st.set_page_config(page_title="AI Medical Compliance Checker", page_icon="🩺", layout="wide")

    st.markdown(
        """
        <style>
            /* General styles */
            .reportview-container {
                background: #ffffff;
            }
            .block-container {
                padding-top: 0rem !important;
                margin-top: 30px !important;
            }
            .stTextArea textarea, .stTextInput input {
                font-size: 16px !important;
                padding: 12px !important;
                border: 1px solid #ccc !important;
                border-radius: 8px !important;
                box-shadow: none !important;
                transition: 0.2s ease-in-out;
                outline: none !important;
            }
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

            /* Mobile Styles */
            @media (max-width: 768px) {
                h1 {
                    font-size: 24px !important;
                    text-align: center;
                }

                /* Button styling for mobile */
                .stButton>button {
                    width: 100% !important;
                    padding: 10px !important;
                    margin-top: 10px !important;
                }

                /* Input field adjustments */
                .stTextArea textarea, .stTextInput input {
                    font-size: 14px !important;
                    padding: 10px !important;
                    width: 100% !important;
                }

                /* Spacing adjustments for inputs */
                .stTextInput, .stTextArea {
                    margin-bottom: 15px !important;
                }

                /* File upload and Admin Key fields */
                .stFileUploader, .stTextInput {
                    width: 100% !important;
                }

                /* Adjust layout for multi-button row */
                .block-container > div {
                    display: flex;
                    flex-direction: column !important;
                }
            }

            /* Desktop and larger screens */
            @media (min-width: 769px) {
                .stButton>button {
                    width: auto !important;
                    margin-top: 20px !important;
                }
            }

        </style>
        """,
        unsafe_allow_html=True
    )


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

# Initialize OpenAI API
client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, connection_timeout=30)
index = pc.Index("compliance-regulations")

# ✅ Extract text from uploaded files (PDF or TXT)
def chunk_text(text, chunk_size=8000):
    """Splits text into smaller chunks to fit within token limits."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text_from_file(file):
    """Extracts text from PDF or TXT file and chunks it to fit OpenAI limits."""
    
    text = ""

    # ✅ Ensure the file is not empty
    file.seek(0)  # Reset pointer
    file_bytes = file.read()  # Read file content
    mime_type, _ = mimetypes.guess_type(file.name)

    if not file_bytes:
        print("⚠️ DEBUG: Uploaded file is empty.")
        return [""]  # Return empty list for chunking compatibility

    # ✅ Process PDF or TXT
    elif mime_type == "application/pdf":
        try:
            file.seek(0)  # Reset again before opening
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in pdf_doc)  # ✅ Process pages one-by-one
        except Exception as e:
            print(f"⚠️ DEBUG: Error reading PDF {file.name}: {e}")
            return [""]

    elif mime_type == "text/plain":
        try:
            text = file_bytes.decode("utf-8")
        except Exception as e:
            print(f"⚠️ DEBUG: Error reading TXT {file.name}: {e}")
            return [""]

    text = text.strip()

    # ✅ Return error if text extraction failed
    if not text:
        print(f"⚠️ DEBUG: No text extracted from {file.name}")
        return [""]

    print(f"✅ DEBUG: Successfully extracted text from {file.name}")

    # ✅ Chunk text if it exceeds max token size
    return chunk_text(text) if len(text) > 8000 else [text]


# ✅ Generate Embeddings for Text Chunks
embedding_cache = {}  # Simple in-memory cache

def create_embedding_with_retry(text_chunks, retries=3, delay=2):
    """Caches embeddings, reduces API calls, and handles retries for batch processing."""
    
    # Step 1: Batch the text chunks
    # If the batch size is too large, OpenAI might reject the request. 
    # So let's break them into manageable sizes.
    batch_size = 50  # This is an arbitrary batch size; adjust based on your needs.
    batches = [text_chunks[i:i + batch_size] for i in range(0, len(text_chunks), batch_size)]

    cached_embeddings = []

    # Step 2: Loop through each batch
    for batch in batches:
        # Check cache for any previously generated embeddings
        batch_embeddings = [embedding_cache.get(chunk) for chunk in batch if chunk in embedding_cache]
        cached_embeddings.extend(batch_embeddings)

        # Get the remaining chunks that don't have cached embeddings
        remaining_chunks = [chunk for chunk in batch if chunk not in embedding_cache]
        
        if remaining_chunks:
            # Retry logic for the batch request
            success = False
            for attempt in range(retries):
                try:
                    # Generate embeddings for the remaining chunks in this batch
                    response = client.embeddings.create(
                        model="text-embedding-3-small", 
                        input=remaining_chunks
                    )
                    
                    if response and "data" in response:
                        embeddings = [item["embedding"] for item in response["data"]]
                        # Cache embeddings
                        for i, chunk in enumerate(remaining_chunks):
                            embedding_cache[chunk] = embeddings[i]
                        cached_embeddings.extend(embeddings)
                        
                        success = True
                        break
                except Exception as e:
                    print(f"⚠️ Error generating embedding for batch. Attempt {attempt + 1}: {e}")
                    time.sleep(delay)
            
            if not success:
                print(f"⚠️ Failed to generate embeddings after {retries} attempts.")
                return []

    return cached_embeddings

policy_cache = {} # Simple dictionary cache
# Assuming `policy_cache` is defined globally or in your class.
# Define a time-to-live (TTL) for cache refresh
CACHE_TTL = 3600  # 1 hour
last_cache_update = {}  # Store the last cache update timestamp

def generate_compliance_policy(topic):
    """Generates a structured compliance policy based on regulations."""
    
    # List of valid compliance-related topics (this could be dynamic if needed)
    valid_topics = ["HIPAA compliance for healthcare", "FDA compliance", "Telehealth regulations", "Data privacy laws", "Patient data security"]
    
    # Check if the entered topic is in the list of valid topics
    if topic not in valid_topics:
        return "⚠️ Please enter a valid compliance-related topic. Example: 'HIPAA compliance for healthcare'."
    
    # Check cache first with TTL consideration
    current_time = time.time()
    if topic in policy_cache and (current_time - last_cache_update.get(topic, 0)) < CACHE_TTL:
        return policy_cache[topic]  # Return cached policy if it's within the TTL

    # Construct the prompt for the policy generation
    prompt = f"""
    Generate a detailed compliance policy for {topic} based on HIPAA, FDA, and other relevant regulations.
    The policy should include the following key sections:
    - **Policy objectives**: What is the purpose of this policy?
    - **Compliance requirements**: What laws or regulations must be followed?
    - **Staff responsibilities**: What roles and responsibilities should staff adhere to?
    - **Risk mitigation measures**: How can potential compliance risks be minimized?
    - **Enforcement strategies**: What actions should be taken to ensure compliance?
    """

    try:
        # Make the API call to generate the compliance policy
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Extract the policy text from the response
        policy_text = response.choices[0].message["content"]
        
        # Cache the policy with the current timestamp
        policy_cache[topic] = policy_text
        last_cache_update[topic] = current_time  # Store the time of the cache update

        return policy_text

    except openai.error.OpenAIError as e:
        # Catch specific OpenAI errors and provide a more meaningful error message
        return f"⚠️ OpenAI is currently unavailable. Please try again later. Error: {e}"
    except Exception as e:
        # Catch other unexpected errors
        return f"⚠️ An error occurred while generating the compliance policy. Error: {e}"
    

# ✅ Store Extracted Text in Pinecone
def store_uploaded_document(file, user_key):
    """Handles document uploads: Admins can store in Pinecone, users get temporary usage."""
    allowed_types = ["application/pdf", "text/plain"]
    mime_type, _ = mimetypes.guess_type(file.name)

    if mime_type not in allowed_types:
        return f"⚠️ Unsupported file type: {mime_type}. Please upload a PDF or TXT file."

    text_chunks = extract_text_from_file(file)  # Get text in chunks
    if not text_chunks:
        return "⚠️ The uploaded document is empty or non-readable. Please upload a valid compliance document."
    
    embeddings = create_embedding_with_retry(text_chunks)  # Generate embeddings for each chunk
    if not embeddings:
        return "⚠️ Failed to process document embedding."

    doc_id = file.name  # Use filename as unique ID
    
    # ✅ Correctly use verify_admin_key(user_key)
    if user_key and verify_admin_key(user_key):  # ✅ Only validate if user enters something
        # ✅ Admin Mode: Store document permanently
        for i, embedding in enumerate(embeddings):
            chunk_id = f"{doc_id}_chunk{i}"
            index.upsert(vectors=[{
                "id": chunk_id,
                "values": embedding,
                "metadata": {"text": text_chunks[i], "filename": doc_id}
            }], namespace="compliance")  # ✅ Ensure the namespace is set
        return f"✅ Document permanently stored in Pinecone: {doc_id}"

    # ✅ Temporary Mode: Store only for session
    return f"⚠️ Temporary Use Only: {doc_id} will not be stored in Pinecone."



def extract_text_and_category(file):
    """Extracts text from a file and categorizes it into HIPAA, FDA, or Telehealth."""
    
    text_chunks = extract_text_from_file(file)  # ✅ Extract as chunks
    text = " ".join(text_chunks)  # ✅ Convert chunks to a single string
    
    # Auto Categorization based on keywords
    category = "Other"
    if "HIPAA" in text or "Privacy Rule" in text or "Breach Notification" in text:
        category = "HIPAA"
    elif "FDA" in text or "drug approval" in text or "clinical trials" in text:
        category = "FDA"
    elif "Telehealth" in text or "remote healthcare" in text or "virtual care" in text:
        category = "Telehealth"

    return text, category  # ✅ Return string, not a list

def verify_admin_key(user_key):
    """Validates the admin key using a direct plain text match."""
    stored_key = os.getenv("ADMIN_SECRET")  # ✅ Fetch stored key (plain text)
    return bool(user_key and stored_key and user_key == stored_key)  # ✅ Direct comparison

def summarize_compliance_information(query, extracted_text, sources):
    """
    Uses OpenAI to summarize retrieved compliance information in a structured format.
    Extracts key points and presents them in a professional way with legal references.
    """

    if not extracted_text:
        return "⚠️ No relevant compliance data retrieved."

    combined_text = " ".join(extracted_text)

    prompt = f"""
    The following compliance-related text has been retrieved from an official compliance database.
    Please summarize the key compliance points in a **structured professional format** that a **Medical Compliance Officer**
    would find useful.

    **Query:** {query}

    **Retrieved Text:**  
    {combined_text}

    **Instructions:**
    - Extract **key compliance rules** relevant to the query.
    - Clearly define **notification deadlines** (e.g., within 60 days of discovery).
    - Break down **enforcement penalties** by severity tiers.
    - Ensure **correct numerical formatting**, e.g.:
      - **$100 per violation, up to $25,000 per year**
      - **$1,000–$50,000 per violation (reasonable cause)**
    - Avoid incorrect tokenization like `"100perviolation,upt025,000"`
    - Use **bold text** instead of italics for emphasis (e.g., `**Tier 1:** $100 per violation`).
    - Ensure **icons are properly spaced** for readability.
    - Ensure **dollar signs and numbers are properly formatted**.

    **Example Output Format:**
    
    ✅ **Key Compliance Points**  
    - Covered entities must comply with **HIPAA Privacy Rule (45 CFR §§ 164.500–534)**  
    - State laws that contradict HIPAA are **preempted by federal requirements**  
    - 💰 **Civil and criminal penalties** apply to HIPAA violations  

    📅 **Notification Deadlines**  
    - **HIPAA requires breach notifications** within **60 days** of discovery  
    - Affected individuals and **OCR must be informed** via written communication  

    🚨 **Enforcement Penalties**  

    **Civil Penalties:**  
    - **Tier 1:** **$100 per violation, up to $25,000 per year**  
    - **Tier 2:** **$1,000–$50,000 per violation** (reasonable cause, corrected violations)  
    - **Tier 3:** **$10,000–$50,000 per violation** (willful neglect, corrected)  
    - **Tier 4:** **$50,000 per violation** (willful neglect, uncorrected; **up to $1.5M per year**)  

    **Criminal Penalties:**  
    - **Fines up to $250,000**  
    - **Prison sentences up to 10 years** for intentional misuse of PHI  

    ✅ **Actionable Steps for Compliance Officers**  
    1️⃣ **Immediately notify affected individuals and OCR**  
    2️⃣ **Conduct forensic audits** to analyze the root cause of the breach  
    3️⃣ 🔒 Implement **corrective security measures** to prevent recurrence  
    4️⃣ 📅 **Train all staff annually** on HIPAA compliance  
    5️⃣ 📝 Maintain **detailed documentation** of all compliance actions  

    ⚠️ **Risks of Noncompliance**  
    - 💰 Financial penalties of **up to $1.5M per year**  
    - ⚖️ **Legal action against executives** for failing to enforce compliance  
    - 💼 **Reputational damage** to the healthcare organization  

    📜 **Legal References**  
    - 📖 **HIPAA Breach Notification Rule (45 CFR §§ 164.400–414)**  
    - 📖 **Office for Civil Rights (OCR) HIPAA Guidelines**  

    📌 **Sources Used:**
    """ + "\n".join(sources)  # ✅ Dynamically include sources

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# Function to generate embeddings with retry logic and exponential backoff
def create_embedding_with_retry(text_chunks, retries=3, delay=2):
    """Caches embeddings, reducing API calls."""
    cached_embeddings = []
    
    for chunk in text_chunks:
        if chunk in embedding_cache:
            cached_embeddings.append(embedding_cache[chunk])  # Reuse existing embedding
        else:
            attempt = 0
            while attempt < retries:
                try:
                    # Generate embedding and store it
                    response = client.embeddings.create(
                        model="text-embedding-3-small", 
                        input=chunk
                    )
                    
                    # Access the response correctly
                    embedding = response.data[0].embedding
                    embedding_cache[chunk] = embedding  # Save for future use
                    cached_embeddings.append(embedding)
                    break  # Exit the retry loop if successful
                except requests.exceptions.Timeout as e:
                    # Handle timeout error
                    print(f"⚠️ Timeout error: {e}. Retrying ({attempt + 1}/{retries})...")
                    attempt += 1
                    if attempt == retries:
                        print("⚠️ Maximum retries reached. Could not generate embedding.")
                except Exception as e:
                    # General exception handling
                    print(f"⚠️ Error generating embedding: {e}")
                    break

    return cached_embeddings


# Function to generate embeddings in bulk
def create_embeddings_in_batch(texts, batch_size=5):
    """Generates embeddings for batches of text chunks to minimize API calls"""
    batched_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batched_embeddings.extend([x['embedding'] for x in response['data']])
        except Exception as e:
            print(f"⚠️ Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
    return batched_embeddings


def search_in_pinecone(query_embedding):
    """Perform optimized query search in Pinecone."""
    try:
        # Perform Pinecone search with vector embedding
        search_results = index.query(
            vector=query_embedding,
            top_k=3,  # Limit results to the top 3 for faster response
            include_metadata=True,
            namespace="compliance"  # Ensure the correct namespace
        )
        return search_results
    except Exception as e:
        print(f"⚠️ Pinecone query error: {e}")
        return None

# ✅ Enhanced Formatting for Output in Streamlit
def search_compliance_documents(query, uploaded_files):
    """
    Searches compliance information in this order:
    1. Pre-computed Pinecone Database (RAG retrieval)
    2. Embedding-based search for documents
    3. OpenAI fallback if no relevant data found
    """
    sources = set()  # Initialize sources here
    extracted_text = []  # Store relevant compliance text
    final_response = ""

    if not query.strip():
        return "⚠️ Please enter a valid compliance question.", [], sources

    # ✅ 1. **Search Pinecone with Precomputed Embeddings**
    query_embedding = create_embedding_with_retry([query])
    if query_embedding:
        search_results = search_in_pinecone(query_embedding[0])

        if search_results and "matches" in search_results:
            for match in search_results["matches"]:
                score = match["score"]
                text = match["metadata"].get("text", "No text found")
                if score >= 0.35:  # threshold
                    sources.add("✅ Retrieved from Compliance Database (Pinecone)")
                    extracted_text.append(text)

            if extracted_text:
                final_response = summarize_compliance_information(query, extracted_text, sources)
                return final_response, extracted_text, sources

    # ✅ 2. **If Pinecone Search Fails, Search Uploaded Documents Using Embeddings**
    if not final_response and uploaded_files:
        document_embeddings = create_embeddings_in_batch([extract_text_from_file(f) for f in uploaded_files])

        for i, embedding in enumerate(document_embeddings):
            search_results = search_in_pinecone(embedding[0])

            if search_results and "matches" in search_results:
                for match in search_results["matches"]:
                    score = match["score"]
                    text = match["metadata"].get("text", "No text found")
                    if score >= 0.35:  # threshold
                        sources.add(f"📄 Retrieved from Attached Document {uploaded_files[i].name}")
                        extracted_text.append(text)

        if extracted_text:
            final_response = summarize_compliance_information(query, extracted_text, sources)
            return final_response, extracted_text, sources

    # ✅ 3. **Fallback to OpenAI (if no results found)**
    if not final_response:
        sources.add("🤖 AI-Generated Answer (No Compliance Data Found)")
        prompt = f"Generate a compliance-related answer for: {query}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            final_response = response.choices[0].message.content
        except Exception as e:
            return f"⚠️ OpenAI error: {e}", [], sources

    # Add source information at the bottom
    formatted_output = f"""
    ✅ **Compliance Answer:**

    📌 **Query:** {query}

    {final_response}

    📌 **Sources Used:**
    """ + "\n".join(sources)  # Dynamically add sources

    return formatted_output, extracted_text, sources

# ✅ Check if Pinecone is Empty
def is_pinecone_empty():
    """Checks if Pinecone has stored documents, handling index errors."""
    try:
        index_stats = index.describe_index_stats()
        total_vectors = index_stats.get("total_vector_count", 0)
        return total_vectors == 0
    except Exception as e:
        print(f"⚠️ Pinecone Index Error: {e}")
        return True  # ✅ Assume empty if an error occurs

    
def save_audit_report(text, filename="Compliance_Audit_Report.txt"):
    """Saves audit reports for compliance records."""
    with open(filename, "w") as file:
        file.write(text)
    return filename

def calculate_risk_score(violations):
    """Assigns a risk score based on violation severity."""
    risk_score = 0
    for v in violations:
        if "High Risk" in v:
            risk_score += 30
        elif "Medium Risk" in v:
            risk_score += 15
        else:
            risk_score += 5
    return min(100, risk_score)  # Cap at 100

def is_compliance_related(question):
    """Checks if the query is related to medical compliance using keywords and Pinecone search."""
    
    # ✅ Step 1: Quick Keyword Check
    compliance_keywords = [
        "HIPAA", "FDA", "medical compliance", "data privacy", "telehealth", 
        "patient data", "healthcare law", "regulatory requirements", "PHI", 
        "privacy rule", "Breach Notification", "security rule", "healthcare compliance"
    ]

    # ✅ If the question contains compliance-related words, return True immediately
    if any(keyword.lower() in question.lower() for keyword in compliance_keywords):
        return True

    # ✅ Step 2: Log failed keyword match (if no match found)
    logging.info(f"⚠️ Failed compliance keyword match for query: '{question}'")

    # ✅ Step 3: Pinecone Search Check (Only if keywords failed)
    if is_pinecone_empty():
        return False  # 🔴 Block unrelated queries if Pinecone is empty

    query_embedding = create_embedding_with_retry([question])
    if not query_embedding:
        return False  # 🔴 Block unrelated queries if embeddings fail

    # ✅ Perform a vector search in Pinecone
    search_results = index.query(vector=query_embedding[0], top_k=3, include_metadata=True)

    if search_results and "matches" in search_results and search_results["matches"]:
        best_match = search_results["matches"][0]

        # ✅ Adjusted similarity threshold to ensure relevance
        if best_match["score"] >= 0.35:
            return True  

    return False  # 🔴 Block unrelated queries



# ✅ Compliance Answer Function
def get_compliance_answer(question, uploaded_files):
    """Retrieves compliance answers from Pinecone first, then attached documents, and lastly OpenAI."""
    
    # Step 1: Search Pinecone
    pinecone_result = search_compliance_documents(question, uploaded_files)

    # Step 2: If no result from Pinecone, search the uploaded documents
    document_results = []
    if not pinecone_result and uploaded_files:  # Only search documents if Pinecone didn't provide a result
        for file in uploaded_files:
            extracted_text = extract_text_from_file(file)
            for chunk in extracted_text:
                if question.lower() in chunk.lower():
                    document_results.append(f"📌 Attached Document: {file.name}\n\n{chunk}")

    # Step 3: If neither Pinecone nor document search found results, use OpenAI
    openai_result = ""
    if not pinecone_result and not document_results:
        prompt = f"""
        No direct compliance regulation was found for the question. Generate an answer using best practices.

        Question: {question}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            openai_result = f"✅ AI-Generated Answer: {response.choices[0].message.content}\n\n(Source: OpenAI GPT)"
        except Exception as e:
            openai_result = f"⚠️ OpenAI is currently unavailable: {e}"

    # Step 4: Compile the final result
    result = "✅ Compliance Information Found:\n\n"
    
    # Add Pinecone result if available
    if pinecone_result:
        result += f"📌 Pinecone: {pinecone_result}\n\n"
    
    # Add Document results if available
    if document_results:
        result += "\n".join(document_results) + "\n\n"
    
    # Add OpenAI result if generated
    if openai_result:
        result += openai_result

    return result

def audit_compliance_document(file):
    """
    Scans a compliance document, extracts missing compliance elements, 
    and generates a structured compliance audit report.
    """

    text_chunks = extract_text_from_file(file)
    full_text = " ".join(text_chunks).strip()  

    if not full_text:
        return "⚠️ This document is empty or non-readable. Please upload a valid compliance policy."

    max_chars = 12000  
    truncated_text = full_text[:max_chars]

    _, category = extract_text_and_category(file)  
    relevant_regulation = search_compliance_documents(category,[])  

    if not relevant_regulation:
        return "⚠️ No compliance regulations found in Pinecone. Please upload relevant compliance documents first."

    audit_prompt = f"""
    ### 🏥 AI Compliance Audit Report
    
    **🔹 Document Category:** {category}  
    **📑 Policy Document (Truncated for AI processing)**  
    ```
    {truncated_text}
    ```
    
    **📜 Relevant Compliance Regulations (From Database)**  
    ```
    {relevant_regulation}
    ```

    ### 📝 **Audit Task**
    1️⃣ Identify **missing compliance elements** in the policy document.  
    2️⃣ Categorize violations into:
       - 🔴 **Privacy Violations** (e.g., patient data breaches)
       - 🔵 **Documentation Errors** (e.g., missing forms, incorrect entries)
       - 🟢 **Patient Safety Risks** (e.g., non-compliance with hygiene policies)
       - 🟠 **Billing Issues** (e.g., Medicare/Medicaid fraud risks)
    3️⃣ Provide **recommendations** for compliance fixes.  
    4️⃣ Summarize key risks and generate a **final risk score (0-100)**.  
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": audit_prompt}],
            temperature=0.3
        )
        audit_result = response.choices[0].message.content
        return audit_result

    except Exception as e:
        return f"⚠️ Error auditing document. Please try again later.\n\n**Error:** {e}"


if not TEST_MODE:
    st.title("🩺 AI Medical Compliance Checker")

    uploaded_files = st.file_uploader(
        "📂 Upload a compliance document (PDF or TXT) - Example: HIPAA, FDA, Telehealth policies",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    user_key = st.text_input("Enter Admin Key (Only for permanent storage)", type="password")
    policy_topic = st.text_input("Enter policy topic (e.g., 'HIPAA compliance for telemedicine')")

    if uploaded_files:
        for uploaded_file in uploaded_files:
           if user_key:  # ✅ Only check admin key if something is entered
            if verify_admin_key(user_key):
                st.success("🔐 Admin mode activated: Storing document in Pinecone.")
            else:
                st.warning("⚠️ Invalid admin key. Document will only be available for this session.")

            # ✅ Step 1: Process the Document Before Storing
            text_chunks = extract_text_from_file(uploaded_file)
            st.write(f"✅ Extracted {len(text_chunks)} chunks from {uploaded_file.name}")

            embeddings = create_embedding_with_retry(text_chunks)
            st.write(f"✅ Generated {len(embeddings)} embeddings for {uploaded_file.name}")

            if embeddings:
                st.write(f"✅ Storing {uploaded_file.name} in Pinecone...")
                store_uploaded_document(uploaded_file, user_key)  # Store the document
                st.success(f"✅ Successfully stored {uploaded_file.name} in Pinecone!")
            else:
                st.error(f"⚠️ Embedding generation failed for {uploaded_file.name}")

    query = st.text_area(
        "Enter your question here:", 
        height=100, 
        placeholder="Example: 'What are the HIPAA compliance requirements for telehealth services?'"
    )
    col1, col2, col3 = st.columns([1, 1, 1])  # Creates three equal-width columns

    with col1:
        if st.button("Check Compliance"):
            if not query.strip():
                st.warning("⚠️ Please enter a compliance-related question.") 
            elif not is_compliance_related(query):  
                st.warning("⚠️ This does not appear to be a compliance-related query. Please ask about compliance regulations.")
            else:
                with st.spinner("🔍 Searching compliance database..."):
                    response, extracted_text, sources = search_compliance_documents(query, uploaded_files)
                    formatted_response = summarize_compliance_information(query, extracted_text, sources) if extracted_text else response

                    # ✅ Display Source Clearly in Streamlit UI
                    st.markdown(formatted_response, unsafe_allow_html=True)

                    st.markdown("### 📌 **Source of Compliance Answer:**")
                    for source in sources:
                        st.markdown(f"- {source}")  # ✅ Shows if it's from Pinecone, Documents, or OpenAI

    with col2:
        if st.button("Audit Compliance"):  
            if uploaded_files and len(uploaded_files) > 0:
                selected_file = uploaded_files[0]  

                if selected_file.getvalue():  
                    selected_file.seek(0)
                    audit_result = audit_compliance_document(selected_file)

                    if "⚠️" in audit_result:
                        st.warning(audit_result)
                    else:
                        st.success(audit_result)
                else:
                    st.error("❌ Uploaded file is empty. Please upload a valid document.")
            else:
                st.warning("⚠️ Please upload a compliance document before auditing.") 

    with col3:
        if st.button("Generate Policy"):
            if policy_topic.strip():
                policy_text = generate_compliance_policy(policy_topic)
                st.success(policy_text)
            else:
                st.warning("⚠️ Please enter a valid topic.")

# Delete all stored vectors from Pinecone
# index.delete(delete_all=True, namespace="compliance")
# print("✅ All stored compliance documents have been deleted from Pinecone.")
