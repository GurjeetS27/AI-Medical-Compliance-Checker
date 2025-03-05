import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("🩺 AI Medical Compliance Checker")
st.write("Ask any medical compliance-related question.")

# User Input
query = st.text_area("Enter your question here:")

# Function to call OpenAI API
def get_compliance_answer(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use gpt-3.5-turbo for better results
        messages=[{"role": "system", "content": "You are a AI Medical Compliance Checker. You will answer any medical compliance-related question."},{"role": "user", "content": question}],
        temperature=0.2,  # Low temperature for factual accuracy
    )
    return response.choices[0].message.content

# Generate Answer
if st.button("Check Compliance"):
    if query.strip():
        answer = get_compliance_answer(query)
        st.subheader("📝 Compliance Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")

