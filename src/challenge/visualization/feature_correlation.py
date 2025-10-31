import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, threshold=0.8):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    high_corr = corr[(corr.abs() > threshold) & (corr.abs() < 1)]
    return high_corr.stack().reset_index().rename(columns={"level_0": "Feature1", "level_1": "Feature2", 0: "Correlation"})

