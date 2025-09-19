def classify_eeg_state(delta, theta, alpha, beta, gamma):
    tbr = theta / beta if beta else 0  # theta/beta ratio
    if tbr > 4.5:
        return "Distracted"
    if beta > alpha + 1.5:
        return "Stressed"
    if alpha > beta + 1.5:
        return "Calm"
    if beta > 5 and gamma > 3:
        return "Focused"
    return "Neutral"
