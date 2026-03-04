import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("movies_2024.csv")

# ------------------------------
# Text Cleaning
# ------------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["Cleaned"] = df["Storyline"].apply(clean_text)

# ------------------------------
# TF-IDF
# ------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Cleaned"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# ------------------------------
# Recommendation Functions
# ------------------------------
def recommend_by_storyline(user_input):
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    scores = list(enumerate(similarity_scores[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = scores[:5]
    return df.iloc[[i[0] for i in top_movies]][["Movie Name", "Storyline"]]

def recommend_by_movie(movie_name):
    index = df[df["Movie Name"] == movie_name].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = scores[1:6]
    return df.iloc[[i[0] for i in top_movies]][["Movie Name", "Storyline"]]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎬 IMDB Movie Recommendation System")

st.markdown("### 🔎 Choose Recommendation Type")

option = st.radio(
    "Select Option",
    ["Recommend by Storyline", "Recommend by Movie Similarity"]
)

# Storyline Input
if option == "Recommend by Storyline":
    user_input = st.text_area("Enter Storyline")

# Movie Dropdown
else:
    movie_name = st.selectbox("Select Movie", df["Movie Name"].values)

# Recommend Button
if st.button("🎯 Recommend"):
    if option == "Recommend by Storyline" and user_input:
        results = recommend_by_storyline(user_input)
    elif option == "Recommend by Movie Similarity":
        results = recommend_by_movie(movie_name)
    else:
        st.warning("Please enter input!")
        st.stop()

    st.markdown("## 🎥 Top 5 Recommended Movies")
    for index, row in results.iterrows():
        st.subheader(row["Movie Name"])
        st.write(row["Storyline"])
        st.markdown("---")