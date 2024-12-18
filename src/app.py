import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

# API endpoint (replace with your FastAPI URL once deployed)
API_URL = "https://semantic-similarity-v3.onrender.com/predict"


# Add an image at the top of the page
st.image("https://apollouniversity.edu.in/blog/wp-content/uploads/2023/03/Biomedical-Sciences.jpg", use_container_width=False)

# Streamlit Title and Description
st.title("Biomedical Sentence Similarity Checker")

st.markdown(
    """
    **This tool calculates the semantic similarity between two biomedical sentences.**
    - Enter two sentences in the fields below.
    - Visualize the similarity score, highlights, and other insights.
    - Similarity score is a continuous value on a scale from 0 to 5, with 0 indicating that the medical semantics
 of the sentences are completely independent and 5 signifying semantic equivalence.  
    """
)

# Input Section
sentence1 = st.text_area("Enter Sentence 1:", height=150)
sentence2 = st.text_area("Enter Sentence 2:", height=150)

# Caching the API response for repeated sentence pairs to avoid redundant calls
@st.cache_data
def get_similarity_score(sentence1, sentence2):
    payload = {"sentence1": sentence1, "sentence2": sentence2}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["similarity_score"]
    else:
        return None

# Button to trigger analysis
if st.button("Analyze Similarity"):
    # Ensure both sentences are provided
    if sentence1 and sentence2:
        # Show a loading spinner
        with st.spinner('Calculating similarity...'):
            # Get similarity score (cached if previously requested)
            similarity_score = get_similarity_score(sentence1, sentence2)

        if similarity_score is not None:
            # Display Similarity Score
            st.success(f"Similarity Score: {similarity_score:.4f} / 5")

            # Visualization
            st.markdown("### Visualizations")

            # Similarity Score Bar Chart
            fig, ax = plt.subplots()
            sns.barplot(x=["Sentence Pair"], y=[similarity_score], palette="Blues_d", ax=ax)
            ax.set_title("Similarity Score Bar Chart")
            ax.set_ylabel("Similarity")
            ax.set_ylim(0, 5)
            st.pyplot(fig)

            # WordCloud Visualization
            st.markdown("### Word Cloud for Sentence 1 and Sentence 2")
            combined_text = f"{sentence1} {sentence2}"
            wordcloud = WordCloud(width=600, height=300, max_words=50, background_color="white").generate(combined_text)

            # Save word cloud image in memory
            img_buffer = BytesIO()
            wordcloud.to_image().save(img_buffer, format="PNG")
            img_buffer.seek(0)
            st.image(img_buffer, caption="Word Cloud", use_container_width=True)

        else:
            st.error("Error computing similarity. Please try again later.")
    else:
        st.warning("Please enter both sentences to compute similarity.")
