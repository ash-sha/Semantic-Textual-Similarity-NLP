# frontend/app.py
import streamlit as st
import requests

# API endpoint URL
API_URL = "http://127.0.0.1:8000/similarity"

# Streamlit layout
st.title("Biomedical Sentence Similarity Checker")

# Input fields for two sentences
sentence1 = st.text_area("Enter Sentence 1:")
sentence2 = st.text_area("Enter Sentence 2:")

# Button to trigger similarity calculation
if st.button("Compute Similarity"):
    # Only call the API if both sentences are provided
    if sentence1 and sentence2:
        # Prepare the request data
        payload = {
            "sentence1": sentence1,
            "sentence2": sentence2
        }

        # Call FastAPI backend to compute similarity
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            similarity_score = response.json()["similarity_score"]
            st.success(f"Similarity Score: {similarity_score:.4f}")
        else:
            st.error("Error in API call. Please try again.")
    else:
        st.warning("Please enter both sentences.")
