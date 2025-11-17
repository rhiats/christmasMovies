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

st.markdown(
    """
    <style>
    .christmas-title {
        color: #B30000;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #FFD700;
        animation: glow 2s ease-in-out infinite alternate;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        font-family: 'Mountains of Christmas', cursive;
    }

    @keyframes glow {
        from { text-shadow: 0 0 5px #FFD700; }
        to { text-shadow: 0 0 25px #FFD700; }
    }
    </style>

    <h1 class="christmas-title">🎄 Christmas Movie Explorer 🎅</h1>
    """,
    unsafe_allow_html=True
)

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
year_range = st.sidebar.slider("Year Range", year_min, year_max, (1900, 2025))

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

#print(df['release_year'])

# Apply filters
filtered = df[df['release_year'].between(*year_range)]
filtered = filtered[filtered['imdb_rating'] >= rating_min]
if selected_genres:
    filtered = filtered[filtered['genres'].apply(lambda x: any(g in x for g in selected_genres))]

#st.write(f"### Showing {len(filtered)} movies")

st.markdown(
    f"<h3 style='color:#B30000;'>Showing {len(filtered)} movies</h3>",
    unsafe_allow_html=True
)

st.dataframe(filtered[['title','release_year','imdb_rating','genres']])

# --- Visualizations ---
#st.write("### 🎬 Movies Per Year")

st.markdown(
    "<h3 style='color:#B30000;'>🎬 Movies Per Year</h3>",
    unsafe_allow_html=True
)

movies_per_year = filtered.groupby('release_year')['title'].count()
fig, ax = plt.subplots()
movies_per_year.plot(kind='bar', ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
st.pyplot(fig)

#st.write("### 📊 Genre Distribution")
st.markdown(
    "<h3 style='color:#B30000;'>📊 Genre Distribution</h3>",
    unsafe_allow_html=True
)
genre_counts = filtered['genres'].explode().value_counts()
fig2, ax2 = plt.subplots()
genre_counts.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Number of Movies")
ax2.set_ylabel("Genre")
st.pyplot(fig2)

# --- Movie Recommendation ---
st.markdown(
    "<h3 style='color:#B30000;'>🎯 Find Similar Movies</h3>",
    unsafe_allow_html=True
)

movie_title = st.selectbox("Select a movie to find similar ones:", filtered['title'].tolist())

# Compute TF-IDF similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Find top 5 similar movies
movie_idx = df[df['title'] == movie_title].index[0]
sim_scores = list(enumerate(cos_sim[movie_idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
top_sim_idx = [i[0] for i in sim_scores[1:6]]  # exclude self

st.write(f"#### Movies similar to **{movie_title}**:")
st.table(df.iloc[top_sim_idx][['title','release_year','rating','genres']])


