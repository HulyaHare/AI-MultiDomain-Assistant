import uuid
from datetime import datetime


class SessionMemory:
    """
    In-memory session store. Keeps conversation history, last intent,
    entities, and engine result for context-aware follow-ups.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self._store[sid] = self._blank()
        return sid

    def get(self, session_id: str) -> dict:
        if session_id not in self._store:
            self._store[session_id] = self._blank()
        return self._store[session_id]

    def update(
        self,
        session_id: str,
        intent: str,
        entities: dict,
        result: dict,
        user_msg: str,
        assistant_msg: str,
    ):
        mem = self.get(session_id)
        mem["last_intent"] = intent
        mem["last_entities"] = entities
        mem["last_result"] = result
        mem["history"].append({"role": "user", "content": user_msg})
        mem["history"].append({"role": "assistant", "content": assistant_msg})
        if len(mem["history"]) > 20:
            mem["history"] = mem["history"][-20:]

    @staticmethod
    def _blank() -> dict:
        return {
            "history": [],
            "last_intent": None,
            "last_entities": None,
            "last_result": None,
            "created_at": datetime.now().isoformat(),
        }


memory = SessionMemory()
