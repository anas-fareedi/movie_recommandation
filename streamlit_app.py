# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# ----------------------
# Tabs for UI
# ----------------------
tab1, tab2 = st.tabs(["Top 10 Recommended Movies", "Find Similar Movies"])

# ----------------------
# Tab 1: Top 10 Recommended
# ----------------------
with tab1:
    st.write("These are the top 10 recommended movies overall:")

    try:
        response = requests.get(" https://movie-recommandation-vypd.onrender.com/predict", timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["Top_Recommended_Movies"])
        df["High_Prob"] = df["High_Prob"].apply(lambda x: round(x, 3))
        df["IMDB_Rating"] = df["IMDB_Rating"].apply(lambda x: round(x, 1))

        st.dataframe(df)

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")

# ----------------------
# Tab 2: Similar Movies
# ----------------------
with tab2:
    st.subheader("Find movies similar to a given title")
    movie_title = st.text_input("Enter movie title:")

    if st.button("Search Similar"):
        if not movie_title:
            st.warning("Please enter a movie title.")
        else:
            try:
                response = requests.get(
                    " https://movie-recommandation-vypd.onrender.com/similar",
                    params={"title": movie_title, "top_k": 10},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                st.write(f"Top similar movies to '{movie_title}':")
                df_sim = pd.DataFrame(data["results"])
                df_sim["similarity_score"] = df_sim["similarity_score"].apply(lambda x: round(x, 3))
                df_sim["IMDB_Rating"] = df_sim["IMDB_Rating"].apply(lambda x: round(x, 1))

                st.dataframe(df_sim)

            except requests.exceptions.RequestException as e:
                st.error(f"Error calling API: {e}")
