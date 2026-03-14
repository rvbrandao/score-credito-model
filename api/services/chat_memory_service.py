from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

MAX_CONTEXT_MESSAGES = 8


@dataclass
class ConversationState:
    turn_count: int = 0
    messages: deque[str] = field(
        default_factory=lambda: deque(maxlen=MAX_CONTEXT_MESSAGES)
    )


_MEMORY: dict[str, ConversationState] = {}
_LOCK = Lock()


def register_user_message(
    message: str,
    conversation_id: str | None,
) -> tuple[str, int, list[str]]:
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Message cannot be empty.")

    resolved_id = (conversation_id or uuid4().hex[:12]).strip()
    if not resolved_id:
        resolved_id = uuid4().hex[:12]

    with _LOCK:
        state = _MEMORY.get(resolved_id)
        if state is None:
            state = ConversationState()
            _MEMORY[resolved_id] = state

        state.turn_count += 1
        state.messages.append(clean_message)

        return resolved_id, state.turn_count, list(state.messages)


def build_context_for_extraction(messages: list[str]) -> str:
    lines = [
        f"Turn {index + 1}: {message}"
        for index, message in enumerate(messages)
    ]
    return "\n".join(lines)
