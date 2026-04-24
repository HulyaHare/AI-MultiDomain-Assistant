import uuid
import math

from fastapi import APIRouter

from app.schemas import QueryRequest, AssistantResponse
from app.services.llm import detect_intent, generate_response
from app.services.memory import memory
from app.engines import healthcare, movie, ecommerce

router = APIRouter(prefix="/api", tags=["assistant"])


def _sanitize(obj):
    """Replace NaN / Inf with None so JSON serialization never breaks."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@router.post("/query", response_model=AssistantResponse)
async def query_assistant(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = memory.get(session_id)

    # 1 — Intent detection via LLM
    intent_data = await detect_intent(req.message, session["history"])
    intent = intent_data.get("intent", "general")
    entities = intent_data.get("entities", {})

    # 2 — Route to the correct domain engine
    engine_result = {}
    try:
        if intent == "healthcare":
            symptoms = entities.get("symptoms", req.message)
            engine_result = healthcare.predict_disease(symptoms)

        elif intent == "movie":
            if entities.get("type") == "similar" and entities.get("title"):
                engine_result = movie.recommend_similar(entities["title"])
            else:
                query = entities.get("query") or entities.get("title") or req.message
                engine_result = movie.search_movies(query)

        elif intent == "ecommerce":
            query = entities.get("query", req.message)
            max_price = entities.get("max_price")
            min_price = entities.get("min_price")
            if isinstance(max_price, str):
                try:
                    max_price = float(max_price)
                except ValueError:
                    max_price = None
            if isinstance(min_price, str):
                try:
                    min_price = float(min_price)
                except ValueError:
                    min_price = None
            engine_result = ecommerce.search_products(
                query, max_price=max_price, min_price=min_price
            )
        else:
            engine_result = {"type": "general"}

    except Exception as e:
        print(f"[Engine] Error in {intent}: {e}")
        engine_result = {"error": str(e)}

    # 3 — LLM generates a natural-language response
    llm_response = await generate_response(intent, req.message, engine_result)

    # 4 — Persist to session memory
    memory.update(
        session_id, intent, entities, engine_result, req.message, llm_response
    )

    return AssistantResponse(
        message=llm_response,
        domain=intent,
        data=_sanitize(engine_result),
        session_id=session_id,
    )
