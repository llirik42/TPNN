import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def draw_heat_map(corr_matrix: pd.DataFrame) -> None:
    sns.heatmap(data=corr_matrix, cmap='coolwarm', annot=True, annot_kws={"fontsize": 10})


def show_all() -> None:
    plt.show()
