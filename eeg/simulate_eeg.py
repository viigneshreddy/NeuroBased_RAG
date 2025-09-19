import random
from datetime import datetime

def simulate_eeg_sample():
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "delta": round(random.uniform(0.5, 3.5), 2),
        "theta": round(random.uniform(2.0, 8.0),  2),
        "alpha": round(random.uniform(3.0, 10.0), 2),
        "beta":  round(random.uniform(1.0, 7.0),  2),
        "gamma": round(random.uniform(0.5, 5.0),  2),
    }

if __name__ == "__main__":
    for _ in range(5):
        print(simulate_eeg_sample())
