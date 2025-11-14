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

df['release_year'] =(
    df['release_year']
    .fillna(0)
    .astype(int) 
    )
year_min = int(df['release_year'].min())
year_max = int(df['release_year'].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (2000, 2025))

# IMDb rating
rating_min = st.sidebar.slider("Minimum IMDb Rating", 0.0, 10.0, 5.0)

# Genre filter
all_genres = df['genres'].explode().unique()
selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

df['genres'] = (
    df['genres']
    .fillna('')                   # replace NaN with empty string
    .apply(lambda x: x if isinstance(x, list) else [])
)

print(df['release_year'])

# Apply filters
filtered = df[df['release_year'].between(*year_range)]
filtered = filtered[filtered['imdb_rating'] >= rating_min]
if selected_genres:
    filtered = filtered[filtered['genres'].apply(lambda x: any(g in x for g in selected_genres))]

st.write(f"### Showing {len(filtered)} movies")
st.dataframe(filtered[['title','release_year','imdb_rating','genres']])

# --- Visualizations ---
st.write("### 🎬 Movies Per Year")
movies_per_year = filtered.groupby('release_year')['title'].count()
fig, ax = plt.subplots()
movies_per_year.plot(kind='bar', ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
st.pyplot(fig)

st.write("### 📊 Genre Distribution")
genre_counts = filtered['genres'].explode().value_counts()
fig2, ax2 = plt.subplots()
genre_counts.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Number of Movies")
ax2.set_ylabel("Genre")
st.pyplot(fig2)


