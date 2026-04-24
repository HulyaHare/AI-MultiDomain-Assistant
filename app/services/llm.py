import json
import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.3,
    max_tokens=1024,
)

# ─── Prompts ──────────────────────────────────────────────────────────────────

INTENT_PROMPT = """You are the intent classifier for a multi-domain AI assistant.
Supported domains:
  1. healthcare — symptoms, diseases, medical concerns
  2. movie      — movie recommendations, actor/genre queries
  3. ecommerce  — product search, shopping, price comparisons

Analyze the user's message (and recent conversation if provided) and return ONLY valid JSON:

{
  "intent": "healthcare" | "movie" | "ecommerce" | "general",
  "entities": { ... }
}

Entity schemas by intent:
  healthcare → {"symptoms": "comma-separated symptom list extracted from user text"}
  movie      → {"title": "movie name or null", "query": "search terms", "type": "similar" or "search"}
               Use "similar" when user references a specific movie title. Use "search" otherwise.
  ecommerce  → {"query": "product search terms", "max_price": number or null, "min_price": number or null}
               NOTE: All prices in this system are Indian Rupees (INR / ₹).
  general    → {"topic": "brief topic description"}

Rules:
- Physical symptoms, feeling sick, health questions → "healthcare"
- Movies, films, actors, genres, directors, watching → "movie"
- Buy, find product, price, shopping, recommend product → "ecommerce"
- If intent is unclear, look at conversation history for context
- Use "general" ONLY for greetings, thanks, or truly unrelated queries
- Output ONLY the JSON object, nothing else"""

RESPONSE_PROMPT = """You are a friendly, knowledgeable AI assistant that provides recommendations
across healthcare, movies, and e-commerce.

Domain: {domain}
User said: {user_message}
Engine results (raw): {engine_result}

Generate a natural, conversational response following these rules:
- Be helpful, warm, and concise (3-5 sentences max for the main text)
- For healthcare: mention the predicted condition, note common symptoms,
  and ALWAYS recommend consulting a real doctor for proper diagnosis
- For movies: present recommendations engagingly, mention ratings and genres
- For ecommerce: highlight top options, mention prices (₹ INR) and key features
- For general: respond naturally as a helpful assistant
- If engine returned empty results or errors, suggest alternative queries
- Do NOT use markdown formatting (no **, no ##, no bullet lists)
- Do NOT repeat raw data — summarize naturally
- Respond in English only"""

GENERAL_PROMPT = """You are a friendly AI assistant that helps with healthcare questions,
movie recommendations, and product shopping. The user sent a general message.
Respond naturally. If their message could relate to health, movies, or shopping,
gently guide them. Keep it brief and conversational. No markdown."""


async def detect_intent(user_message: str, history: list) -> dict:
    history_text = ""
    if history:
        recent = history[-6:]
        history_text = "\n\nRecent conversation:\n" + "\n".join(
            f"  {m['role']}: {m['content'][:120]}" for m in recent
        )

    messages = [
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=f"User message: {user_message}{history_text}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[LLM] Intent detection error: {e}")

    return {"intent": "general", "entities": {"topic": user_message}}


async def generate_response(
    domain: str, user_message: str, engine_result: dict
) -> str:
    if domain == "general":
        messages = [
            SystemMessage(content=GENERAL_PROMPT),
            HumanMessage(content=user_message),
        ]
    else:
        result_str = json.dumps(engine_result, default=str, ensure_ascii=False)[:2000]
        messages = [
            SystemMessage(
                content=RESPONSE_PROMPT.format(
                    domain=domain,
                    user_message=user_message,
                    engine_result=result_str,
                )
            ),
            HumanMessage(content="Generate the response now."),
        ]

    try:
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"[LLM] Response generation error: {e}")
        return _fallback(domain, engine_result)


def _fallback(domain: str, result: dict) -> str:
    if domain == "healthcare":
        d = result.get("predicted_disease", "a condition")
        return f"Based on your symptoms, this could be related to {d}. Please consult a healthcare professional for a proper diagnosis."
    if domain == "movie":
        movies = result.get("movies", [])
        if movies:
            titles = ", ".join(m["title"] for m in movies[:3])
            return f"Here are some recommendations: {titles}. Check out the details below!"
        return "I couldn't find matching movies. Could you try a different title or describe what you're looking for?"
    if domain == "ecommerce":
        products = result.get("products", [])
        if products:
            names = ", ".join(p["name"] for p in products[:3])
            return f"I found some options for you: {names}. See the details below!"
        return "No matching products found. Try different search terms or adjust your price range."
    return "Hello! I can help you with health questions, movie recommendations, or product shopping. What would you like?"
