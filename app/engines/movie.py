import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.data.loader import DataStore


def _row_to_dict(row, similarity: float = None) -> dict:
    return {
        "title": row["title"],
        "overview": str(row.get("overview", ""))[:250],
        "genres": row.get("genre_names", "").replace("  ", " ").strip(),
        "vote_average": float(row.get("vote_average", 0)),
        "release_date": str(row.get("release_date", "")),
        "cast": row.get("cast_names", "").replace("  ", " ").strip()[:120],
        "similarity": round(float(similarity), 3) if similarity is not None else None,
    }


def recommend_similar(title: str, top_n: int = 8) -> dict:
    """Find movies similar to a given title using cosine similarity on TF-IDF vectors."""
    df = DataStore.movies_df

    mask = df["title"].str.lower() == title.strip().lower()
    if not mask.any():
        mask = df["title"].str.lower().str.contains(title.strip().lower(), na=False)

    if not mask.any():
        return {"movies": [], "message": f"Movie '{title}' not found. Try a different title or describe what you want."}

    idx = mask.idxmax()
    movie_vec = DataStore.movie_tfidf_matrix[idx]
    sims = cosine_similarity(movie_vec, DataStore.movie_tfidf_matrix).flatten()

    sims[idx] = -1
    top_indices = sims.argsort()[-top_n:][::-1]

    results = [_row_to_dict(df.iloc[i], sims[i]) for i in top_indices]
    return {"movies": results, "reference": df.iloc[idx]["title"]}


def search_movies(query: str, top_n: int = 8) -> dict:
    """Search movies by free-text query (genre, keyword, description, actor, etc.)."""
    query_vec = DataStore.movie_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, DataStore.movie_tfidf_matrix).flatten()

    top_indices = sims.argsort()[-top_n:][::-1]
    df = DataStore.movies_df

    results = [_row_to_dict(df.iloc[i], sims[i]) for i in top_indices if sims[i] >= 0.01]
    return {"movies": results, "query": query}
