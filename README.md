# 🧠 NeuroBased RAG — LLMs Guided by Brain Activity

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![EEG](https://img.shields.io/badge/EEG-Muse%20%2F%20NeuroSky-8b5cf6?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented-10b981?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-HuggingFace-f59e0b?style=for-the-badge&logo=huggingface&logoColor=white)
![Stars](https://img.shields.io/github/stars/viigneshreddy/NeuroBased_RAG?style=for-the-badge)

## Overview

**NeuroBased RAG** is an innovative system that integrates real-time **EEG (electroencephalogram) brain signals** with **Large Language Models (LLMs)** through a **Retrieval-Augmented Generation (RAG)** pipeline. The system adaptively tailors its responses based on the user's live cognitive state — making it especially powerful for **cognitive therapy** applications targeting conditions such as **ADHD** and **PTSD**.

By coupling affordable consumer EEG devices (Muse, NeuroSky) with local LLMs and a brain-signal classifier, the system dynamically adjusts retrieval depth, prompt tone, and response complexity in real time — meeting users where their brain actually is.

---

## Key Idea

> *"What if an AI could sense when you're overwhelmed, distracted, or anxious — and respond accordingly?"*

Traditional LLMs treat every user interaction identically. NeuroBased RAG breaks that pattern: EEG-derived cognitive state signals (focus, stress, relaxation) feed directly into the RAG pipeline's prompt modifier, making every response adaptive and clinically aware.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        NeuroBased RAG                    │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │  EEG Device │───▶│  Classifier  │───▶│   Prompt    │ │
│  │ (Muse/Neuro)│    │ (classifier  │    │  Modifier   │ │
│  └─────────────┘    │    .py)      │    │(prompt_mod  │ │
│                     └──────────────┘    │  ifier.py)  │ │
│  ┌─────────────┐                        └──────┬──────┘ │
│  │  EEG Signal │                               │        │
│  │  Simulator  │                               ▼        │
│  │(simulate_ee │             ┌─────────────────────────┐│
│  │   g.py)     │             │      RAG Pipeline       ││
│  └─────────────┘             │   (rag_pipeline.py)     ││
│                              │  ┌─────────────────────┐││
│                              │  │  Retrieval Engine   │││
│                              │  │  (retrieval.py)     │││
│                              │  └─────────────────────┘││
│                              └──────────┬──────────────┘│
│                                         │               │
│                                         ▼               │
│                     ┌───────────────────────────────┐   │
│                     │  LLM Provider (hf_local.py)   │   │
│                     │  HuggingFace Local Inference   │   │
│                     └───────────────────────────────┘   │
│                                         │               │
│                                         ▼               │
│                          ┌──────────────────────┐       │
│                          │   Streamlit UI        │       │
│                          │    (ui/app.py)        │       │
│                          └──────────────────────┘       │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
NeuroBased_RAG/
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
│
├── eeg/
│   ├── classifier.py         # EEG signal classification (focus / stress / relaxed states)
│   └── simulate_eeg.py       # Synthetic EEG signal generator for testing
│
├── llm/
│   └── providers/
│       └── hf_local.py       # Local HuggingFace model inference provider
│
├── rag/
│   ├── rag_pipeline.py       # Core RAG orchestration — ties retrieval + LLM + EEG state
│   ├── retrieval.py          # Document retrieval from knowledge base
│   └── prompt_modifier.py    # Dynamically adapts prompts based on EEG cognitive state
│
└── ui/
    └── app.py                # Streamlit / Flask web interface
```

---

## Module Descriptions

### `eeg/classifier.py`
Processes raw EEG signal data and classifies the user's cognitive state into categories such as **focused**, **stressed**, or **relaxed**. This state is passed downstream to the prompt modifier to influence how the LLM responds.

### `eeg/simulate_eeg.py`
Generates synthetic EEG signals for development and testing without requiring a physical EEG headset. Useful for CI pipelines and demos.

### `llm/providers/hf_local.py`
Wraps a locally-hosted HuggingFace language model, enabling fully offline inference — critical for privacy-sensitive therapeutic applications where patient data must not leave the device.

### `rag/retrieval.py`
Implements the document retrieval engine. Given a query, it fetches the most relevant context chunks from a knowledge base (e.g., therapeutic guidelines, CBT frameworks, PTSD literature).

### `rag/prompt_modifier.py`
The brain of the adaptive system. Takes the classified EEG state and the user's query, then reshapes the LLM prompt accordingly — simplifying language for high-stress states, deepening detail for high-focus states, or adding grounding techniques for anxious states.

### `rag/rag_pipeline.py`
Orchestrates the full pipeline: receives user input + EEG state → retrieves relevant documents → modifies prompt → calls the LLM → returns a cognitively-adapted response.

### `ui/app.py`
Provides a browser-based interface for interacting with the system, visualizing EEG state in real-time, and reading adaptive LLM responses.

---

## Getting Started

### Prerequisites

- Python 3.9+
- A compatible EEG device (Muse 2, NeuroSky MindWave) **or** use the built-in simulator
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/viigneshreddy/NeuroBased_RAG.git
cd NeuroBased_RAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Run the main application
python main.py

# Or launch the UI directly
python ui/app.py
```

### Run with Simulated EEG (no hardware needed)

```bash
# The simulator generates synthetic brain signal data
python eeg/simulate_eeg.py
```

---

## Use Cases

| Condition | How NeuroBased RAG Helps |
|-----------|--------------------------|
| **ADHD** | Detects low-focus states and simplifies responses; uses shorter, structured prompts |
| **PTSD** | Detects high-stress / hyperarousal states and shifts to grounding language |
| **Anxiety** | Adapts tone to be calming and avoids overwhelming information density |
| **General Cognitive Therapy** | Tracks session-level cognitive patterns for clinician review |

---

## Why Local LLMs?

Patient data is highly sensitive. By running HuggingFace models **locally via `hf_local.py`**, all EEG signals, conversation history, and therapy content stay on-device — fully compliant with privacy requirements and suitable for clinical environments without internet access.

---

## Roadmap

- [ ] Real-time EEG stream integration (Muse SDK / OpenBCI)
- [ ] Multi-LLM provider support (Ollama, OpenAI, Anthropic)
- [ ] Session history and cognitive state logging for clinicians
- [ ] Fine-tuned therapy-specific retrieval corpus
- [ ] Mobile-friendly UI
- [ ] Clinical trial integration

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## Disclaimer

This project is a research prototype and is **not a certified medical device**. It is not intended to replace professional clinical diagnosis or therapy. Always consult a qualified mental health professional for medical advice.

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

---

*Bridging neuroscience and AI — one brain signal at a time.*
