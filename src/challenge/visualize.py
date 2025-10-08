import matplotlib.pyplot as plt
import numpy as np

def plot_signal(x: np.ndarray, title: str = "Signal"):
    plt.figure()
    plt.plot(x)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    return plt.gca()
