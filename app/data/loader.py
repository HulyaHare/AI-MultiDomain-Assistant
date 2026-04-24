import json
import ast
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataStore:
    """Singleton-style global data store populated once during FastAPI lifespan."""

    # Healthcare
    health_df: pd.DataFrame = None
    health_vectorizer: TfidfVectorizer = None
    health_model: LinearSVC = None
    health_label_encoder: LabelEncoder = None

    # Movies
    movies_df: pd.DataFrame = None
    movie_tfidf_matrix = None
    movie_vectorizer: TfidfVectorizer = None

    # E-commerce
    products_df: pd.DataFrame = None
    product_tfidf_matrix = None
    product_vectorizer: TfidfVectorizer = None


def _safe_json_parse(val):
    """Parse JSON-like string fields from TMDB data safely."""
    if pd.isna(val) or val == "":
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return []


def _extract_names(items, key="name", limit=None):
    """Extract name values from a list of dicts, joining with spaces (no whitespace in names)."""
    if not items:
        return ""
    subset = items[:limit] if limit else items
    return " ".join(item.get(key, "").replace(" ", "") for item in subset if isinstance(item, dict))


# ─── Healthcare ───────────────────────────────────────────────────────────────

def load_healthcare():
    path = os.path.join(BASE_DIR, "Healthcare.csv")
    df = pd.read_csv(path)
    df["Symptoms"] = df["Symptoms"].fillna("")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["Symptoms"])

    le = LabelEncoder()
    y = le.fit_transform(df["Disease"].fillna("Unknown"))

    model = LinearSVC(max_iter=10000, C=1.0)
    model.fit(X, y)

    DataStore.health_df = df
    DataStore.health_vectorizer = vectorizer
    DataStore.health_model = model
    DataStore.health_label_encoder = le
    print(f"  [Healthcare] {len(df)} records, {len(le.classes_)} diseases")


# ─── Movies ───────────────────────────────────────────────────────────────────

def load_movies():
    movies_path = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")
    credits_path = os.path.join(BASE_DIR, "tmdb_5000_credits.csv")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    movies = movies.merge(
        credits, left_on="id", right_on="movie_id", how="left", suffixes=("", "_cr")
    )
    if "title_cr" in movies.columns:
        movies.drop(columns=["title_cr"], inplace=True)

    movies["genres_list"] = movies["genres"].apply(_safe_json_parse)
    movies["keywords_list"] = movies["keywords"].apply(_safe_json_parse)
    movies["cast_list"] = movies["cast"].apply(_safe_json_parse)
    movies["crew_list"] = movies["crew"].apply(_safe_json_parse)

    movies["genre_names"] = movies["genres_list"].apply(lambda x: _extract_names(x))
    movies["keyword_names"] = movies["keywords_list"].apply(lambda x: _extract_names(x, limit=15))
    movies["cast_names"] = movies["cast_list"].apply(lambda x: _extract_names(x, limit=5))
    movies["director_name"] = movies["crew_list"].apply(
        lambda crew: " ".join(
            c["name"].replace(" ", "") for c in crew if isinstance(c, dict) and c.get("job") == "Director"
        ) if crew else ""
    )

    for col in ["overview", "genre_names", "keyword_names", "cast_names", "director_name", "title"]:
        movies[col] = movies[col].fillna("")

    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0)
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0)

    movies["combined"] = (
        movies["genre_names"] + " " +
        movies["keyword_names"] + " " +
        movies["cast_names"] + " " +
        movies["director_name"] + " " +
        movies["overview"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["combined"])

    DataStore.movies_df = movies
    DataStore.movie_tfidf_matrix = tfidf_matrix
    DataStore.movie_vectorizer = vectorizer
    print(f"  [Movies] {len(movies)} movies loaded")


# ─── E-commerce ───────────────────────────────────────────────────────────────

def load_ecommerce():
    path = os.path.join(BASE_DIR, "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(path)

    df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce")
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
    df["product_rating"] = pd.to_numeric(df["product_rating"], errors="coerce")
    df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")

    for col in ["product_name", "description", "product_category_tree", "brand"]:
        df[col] = df[col].fillna("")

    df["category_clean"] = df["product_category_tree"].apply(
        lambda x: " >> ".join(x.strip('[]"').split(">>")[:3]).strip() if isinstance(x, str) and x else ""
    )

    df["desc_short"] = df["description"].str[:500]

    df["search_text"] = (
        df["product_name"] + " " +
        df["desc_short"] + " " +
        df["category_clean"] + " " +
        df["brand"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])

    DataStore.products_df = df
    DataStore.product_tfidf_matrix = tfidf_matrix
    DataStore.product_vectorizer = vectorizer
    print(f"  [E-commerce] {len(df)} products loaded")
