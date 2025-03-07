import streamlit as st
import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load API Keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI API
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # ✅ Updated for latest OpenAI API

# Initialize Pinecone
pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("compliance-regulations")  # Ensure this matches your Pinecone index name

# Predefined compliance regulations for storage
COMPLIANCE_DOCUMENTS = [
    {"id": "HIPAA-001", "text": "Under HIPAA, patients have the right to access their medical records within 30 days of a request."},
    {"id": "HIPAA-002", "text": "HIPAA requires healthcare providers to keep patient data confidential and secure."},
    {"id": "HIPAA-003", "text": "Healthcare providers must obtain written consent before sharing patient information with third parties."},
    {"id": "FDA-001", "text": "The FDA mandates that all medical devices undergo rigorous testing before market approval."},
    {"id": "FDA-002", "text": "Pharmaceutical companies must conduct clinical trials before a drug receives FDA approval."}
]

# ✅ Convert Text to Embeddings using OpenAI (Latest Version)
def create_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]  # ✅ OpenAI now requires input as a list
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# ✅ Store Compliance Documents in Pinecone
def store_compliance_documents():
    for doc in COMPLIANCE_DOCUMENTS:
        embedding = create_embedding(doc["text"])
        if embedding:
            index.upsert(vectors=[{"id": doc["id"], "values": embedding, "metadata": {"text": doc["text"]}}])
            print(f"✅ Stored document {doc['id']} in Pinecone.")

# ✅ Search for Compliance Documents in Pinecone
def search_compliance_documents(query):
    query_embedding = create_embedding(query)
    if not query_embedding:
        return None  # Handle case where embedding generation fails

    search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Debugging: Print Pinecone search results
    print("\n🔍 Pinecone Search Results:", search_results)

    # Extract the best-matching compliance document
    if search_results and search_results["matches"]:
        best_match = search_results["matches"][0]["metadata"]["text"]
        print(f"\n✅ Using Pinecone Result: {best_match}")
        return best_match

    print("\n❌ No relevant result found in Pinecone, using OpenAI only.")
    return None  # If no match is found, return None

# ✅ Get Compliance Answer (Uses Pinecone + OpenAI)
def get_compliance_answer(question):
    relevant_document = search_compliance_documents(question)

    if relevant_document:
        prompt = f"Based on this compliance regulation: {relevant_document}, answer the question: {question}"
        source = "Pinecone ✅"
    else:
        prompt = question
        source = "OpenAI Only ❌"

    try:
        response = client.chat.completions.create(  # ✅ Updated OpenAI API call
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        final_answer = response.choices[0].message.content
    except Exception as e:
        final_answer = "⚠️ Error generating response."
        print(f"❌ OpenAI Error: {e}")
        source = "Error ❌"

    print(f"\n🔹 Answer Generated Using: {source}")
    return f"{final_answer}\n\n(Source: {source})"

# ✅ Streamlit UI
st.title("🩺 AI Medical Compliance Checker")
st.write("Ask any medical compliance-related question.")

query = st.text_area("Enter your question here:")

if st.button("Check Compliance"):
    if query.strip():
        answer = get_compliance_answer(query)
        st.subheader("📝 Compliance Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please enter a question.")

# ✅ Run Once: Store compliance documents in Pinecone (Uncomment to run manually)
# store_compliance_documents()
