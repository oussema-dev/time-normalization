import pandas as pd
import matplotlib.pyplot as plt

"""
Note: when plotting data with different column names, it will result in them plotting next to each other instead of being superimposed
"""


def plot_first_row(df1, df2):
    plt.plot(df1.columns, df1.iloc[0], label="Processed data")
    plt.plot(df2.columns, df2.iloc[0], label="Raw data")
    plt.xlabel("Frames")
    plt.ylabel("Amplitudes")
    plt.title("First trial Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file1 = "data\\processed data\\GRF_F_V_PRO_right.csv"
    file2 = "data\\raw data\\GRF_F_V_RAW_right.csv"

    df1 = pd.read_csv(file1)
    df1 = df1.iloc[:, 4:-2]

    df2 = pd.read_csv(file2)
    df2 = df2.iloc[:, 4:-2]

    plot_first_row(df1, df2)
