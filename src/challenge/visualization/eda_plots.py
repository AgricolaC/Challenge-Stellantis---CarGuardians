import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def eda_plots_with_stats(df: pd.DataFrame, title_prefix: str = ""):
    """
    Prints mean/std by class (0/1) and plots:
      - PDF (KDE) of each feature for both classes
      - CDF (KDE with cumulative=True)
      - Box plot (feature vs class)

    Works directly on data that may include missing values.
    """
    sns.set_theme(style="whitegrid")
    features = [c for c in df.columns if c != "class"]

    # Pre-calc describes per class to mimic teacher
    desc_0 = df[df["class"] == 0].describe(include="all")
    desc_1 = df[df["class"] == 1].describe(include="all")

    for feat in features:
        # Print means/std like teacher
        if feat in desc_1.columns and feat in desc_0.columns:
            mean1 = desc_1.loc["mean", feat] if "mean" in desc_1.index else np.nan
            std1 = desc_1.loc["std", feat] if "std" in desc_1.index else np.nan
            mean0 = desc_0.loc["mean", feat] if "mean" in desc_0.index else np.nan
            std0 = desc_0.loc["std", feat] if "std" in desc_0.index else np.nan

            print(
                f"\033[1mFeature '{feat}', Class 1 → Mean: {mean1:.4f}, Std: {std1:.4f}\033[0m"
            )
            print(f"Feature '{feat}', Class 0 → Mean: {mean0:.4f}, Std: {std0:.4f}")
        else:
            print(f"(Skipping stats for '{feat}' — missing in describe tables)")

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{title_prefix}{feat}", fontsize=13)

        # PDF (KDE) — handle NaNs automatically
        try:
            sns.kdeplot(data=df, x=feat, hue="class", common_norm=False, ax=ax[0])
            ax[0].set_title("PDF (KDE)")
        except Exception as e:
            ax[0].set_title("PDF (KDE) — error")
            ax[0].text(0.5, 0.5, str(e), ha="center")

        # CDF
        try:
            sns.kdeplot(
                data=df,
                x=feat,
                hue="class",
                common_norm=False,
                cumulative=True,
                ax=ax[1],
            )
            ax[1].set_title("CDF (KDE cumulative)")
        except Exception as e:
            ax[1].set_title("CDF — error")
            ax[1].text(0.5, 0.5, str(e), ha="center")

        # Box
        try:
            sns.boxplot(data=df, x="class", y=feat, ax=ax[2])
            ax[2].set_title("Box Plot")
            ax[2].set_xlabel("class (0=neg, 1=pos)")
        except Exception as e:
            ax[2].set_title("Box — error")
            ax[2].text(0.5, 0.5, str(e), ha="center")

        plt.tight_layout()
        plt.show()
        print("*" * 100)
