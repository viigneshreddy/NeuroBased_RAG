from collections import deque

PROMPT_TEMPLATES = {
    "Stressed":   "Be concise, warm, and soothing. Offer 3 short steps (breathing, grounding, reframing).",
    "Focused":    "Go deeper and structured. Provide 3–5 detailed steps with rationale.",
    "Calm":       "Maintain a gentle tone. Suggest light mindfulness or consolidation tips.",
    "Distracted": "Use short, motivating bullets. One tiny task now, then a simple next step.",
    "Neutral":    "Offer balanced, supportive guidance with clear, actionable steps."
}

def last_n_profile(eeg_states, n=5):
    dq = deque(eeg_states[-n:], maxlen=n)
    if dq.count("Stressed") >= 3: return "Stressed"
    if dq.count("Focused")  >= 3: return "Focused"
    return dq[-1] if dq else "Neutral"

def modify_prompt(query, context, eeg_state, eeg_history=None):
    profile = last_n_profile(eeg_history or []) or eeg_state or "Neutral"
    instruction = PROMPT_TEMPLATES.get(profile, PROMPT_TEMPLATES["Neutral"])
    return f"""You are a supportive therapy copilot. Do not diagnose; give safe, practical guidance.

[User Query]
{query}

[Retrieved Context]
{context}

[EEG-informed Style]
{instruction}

[Output Rules]
- Ground advice in the retrieved context when possible.
- Use numbered steps and keep to ~120–180 words.
- Keep language empathetic and clear.
"""

