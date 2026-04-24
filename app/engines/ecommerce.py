import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.data.loader import DataStore


def search_products(
    query: str,
    max_price: float = None,
    min_price: float = None,
    top_n: int = 8,
) -> dict:
    """
    Search products by text similarity, optionally filtered by price range.
    Prices are in INR (Indian Rupees) from the Flipkart dataset.
    """
    query_vec = DataStore.product_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, DataStore.product_tfidf_matrix).flatten()

    df = DataStore.products_df
    filtered_sims = sims.copy()

    # Price filtering uses discounted_price (falls back to retail_price)
    effective_price = df["discounted_price"].fillna(df["retail_price"])
    if max_price is not None:
        filtered_sims[effective_price.gt(max_price).fillna(False).values] = -1
    if min_price is not None:
        filtered_sims[effective_price.lt(min_price).fillna(False).values] = -1

    top_indices = filtered_sims.argsort()[-top_n:][::-1]

    results = []
    for i in top_indices:
        if filtered_sims[i] < 0.01:
            continue
        row = df.iloc[i]
        rp = row["retail_price"]
        dp = row["discounted_price"]
        pr = row["product_rating"]
        results.append({
            "name": row["product_name"],
            "retail_price": float(rp) if not np.isnan(rp) else None,
            "discounted_price": float(dp) if not np.isnan(dp) else None,
            "rating": float(pr) if not np.isnan(pr) else None,
            "brand": row.get("brand", "") or "",
            "category": row.get("category_clean", "")[:150],
            "description": str(row.get("description", ""))[:250],
            "similarity": round(float(filtered_sims[i]), 3),
        })

    return {"products": results, "query": query}
