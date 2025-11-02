import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("christmas_movies.csv")
    df['genres'] = df['genre'].str.split(', ')
    df['description'] = df['description'].fillna('')
    return df

df = load_data()

st.title("🎄 Christmas Movie Explorer")

# --- Sidebar filters ---
st.sidebar.header("Filter Movies")

# Year range
year_min = int(df['release_year'].min())
year_max = int(df['release_year'].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (2000, 2025))
