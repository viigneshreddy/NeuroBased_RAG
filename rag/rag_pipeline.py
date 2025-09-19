from rag.retrieval import hybrid_retrieve
from typing import Optional, List
from rag.prompt_modifier import modify_prompt
from llm.providers.hf_local import generate as hf_generate

DOCUMENTS = [
    "Close your eyes and take five deep breaths.",
    "Stay hydrated and take a short walk every hour.",
    "Try using the Pomodoro technique to stay focused.",
    "Practice grounding: name 5 things you can see and feel.",
    "Visualize your success. Positive reinforcement builds confidence."
]

def generate_response(query: str, eeg_state: str, eeg_history: Optional[List[str]] = None) -> str:
    retrieved = hybrid_retrieve(query, DOCUMENTS, top_k=3)
    context = "\n".join(f"- {d}" for d in retrieved)
    prompt = modify_prompt(query, context, eeg_state, eeg_history or [])
    return hf_generate(prompt)
