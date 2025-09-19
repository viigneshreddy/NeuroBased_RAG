import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

from eeg.simulate_eeg import simulate_eeg_sample
from eeg.classifier import classify_eeg_state
from rag.rag_pipeline import generate_response

def demo_once():
    eeg = simulate_eeg_sample()
    eeg_state = classify_eeg_state(**{k: eeg[k] for k in ["delta","theta","alpha","beta","gamma"]})
    query = "I have a big exam tomorrow and I'm feeling overwhelmed. How can I calm down and focus?"
    answer = generate_response(query, eeg_state, eeg_history=[eeg_state]*3)

    print("EEG sample:", eeg)
    print("EEG state:", eeg_state)
    print("\n--- Answer ---\n")
    print(answer)

if __name__ == "__main__":
    demo_once()
