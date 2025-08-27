# from fastapi import FastAPI
# import joblib
# import pandas as pd

# app = FastAPI()

# cosine_sim = joblib.load("cosine_sim.pkl")
# indices = joblib.load("indices.pkl")
# df = pd.read_csv("movies_imdb.csv")

# @app.get("/")
# def home():
#     return {"message": "Movie Recommendation API is running"}

# @app.get("/recommend/")
# def recommend_api(movie_title: str):
#     if movie_title not in indices:
#         return {"error": "Movie not found in dataset"}
    
#     idx = indices[movie_title]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
#     movie_indices = [i[0] for i in sim_scores]
#     recommendations = df["Series_Title"].iloc[movie_indices].tolist()
    
#     return {"input": movie_title, "recommendations": recommendations}\

# ABOVE IS NON TOUCHABLE CODE (its for the non trained model like : cosine_sim , Indices.pkl )

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = None
tfidf = None
movies_df = None
tfidf_matrix = None

class PredictRequest(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None


@app.on_event("startup")
def load_artifacts():
    global clf, tfidf, movies_df, tfidf_matrix

    try:
        clf = joblib.load("rf_model.joblib")
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        movies_df = joblib.load("movies_df.pkl")
    except Exception as e:
        raise RuntimeError(f"Error loading artifacts: {e}")

    tfidf_matrix = tfidf.transform(movies_df["combined_features"].astype(str))


@app.get("/")
def root():
    return {"msg": "Movie Recommender API. Visit /docs for interactive API."}


# @app.post("/predict")
# def predict(req: PredictRequest):
#     if clf is None or tfidf is None:
#         raise HTTPException(500, "Model or vectorizer not loaded")

#     combined = " ".join(filter(None, [req.title, req.genre])).strip()
#     if not combined:
#         raise HTTPException(400, "Send at least one of title/genre")

#     movies_df["High_Prob"] = clf.predict_proba(tfidf_matrix)[:, 1]
#     top_movies = movies_df.sort_values("High_Prob", ascending=False).head(10)

#     results = top_movies[["Series_Title", "Genre", "IMDB_Rating", "High_Prob"]] \
#         .to_dict(orient="records")

#     return {
#         "input": {"title": req.title, "genre": req.genre},
#         "Top_Recommended_Movies": results
#     }

@app.get("/predict")
def predict():
    global clf, tfidf, movies_df

    if clf is None or movies_df is None or tfidf is None:
        raise HTTPException(500, "Model, vectorizer, or data not loaded")

    # Transform combined_features using tfidf
    X = tfidf.transform(movies_df["combined_features"].astype(str))

    # Predict probabilities
    if hasattr(clf, "predict_proba"):
        movies_df["High_Prob"] = clf.predict_proba(X)[:, 1]

    top_movies = movies_df.sort_values("High_Prob", ascending=False).head(10)
    results = top_movies[["Series_Title", "Genre", "IMDB_Rating", "High_Prob"]].to_dict(orient="records")

    return {"Top_Recommended_Movies": results}



@app.get("/similar")
def similar(title: str = Query(..., description="Movie title to find similar movies"),
            top_k: int = Query(5, ge=1, le=50)):
    if tfidf is None or tfidf_matrix is None:
        raise HTTPException(500, "Vectorizer or matrix not available")

    mask = movies_df["Series_Title"].str.lower() == title.lower()
    if mask.any():
        idx = movies_df[mask].index[0]
        vec = tfidf_matrix[idx]
        sims = cosine_similarity(vec, tfidf_matrix).flatten()
        sims[idx] = -1  # exclude itself
    else:
        # Use query title as text
        vec = tfidf.transform([title])
        sims = cosine_similarity(vec, tfidf_matrix).flatten()

    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "Series_Title": movies_df.loc[i, "Series_Title"],
            "Genre": movies_df.loc[i, "Genre"],
            "IMDB_Rating": float(movies_df.loc[i, "IMDB_Rating"]),
            "similarity_score": float(sims[i])
        })
    return {"query_title": title, "results": results}