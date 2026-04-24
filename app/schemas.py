from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class MovieItem(BaseModel):
    title: str
    overview: str
    genres: str
    vote_average: float
    release_date: Optional[str] = None
    cast: Optional[str] = None
    similarity: Optional[float] = None


class ProductItem(BaseModel):
    name: str
    retail_price: Optional[float] = None
    discounted_price: Optional[float] = None
    rating: Optional[float] = None
    brand: Optional[str] = None
    category: str
    description: str
    similarity: Optional[float] = None


class HealthResult(BaseModel):
    predicted_disease: str
    confidence: Optional[float] = None
    sample_cases: Optional[list[str]] = None
    total_cases_in_data: Optional[int] = None


class AssistantResponse(BaseModel):
    message: str
    domain: str
    data: Optional[dict] = None
    session_id: str
