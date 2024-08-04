import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(__file__)

dir = "data"
subdir = "freddy_uc1_test"
save_dir = f"plots/{subdir}"
file_name1 = "kinova_left_voltage_current_log.csv"
file_name2 = "kinova_right_voltage_current_log.csv"
file_name3 = "mobile_base_voltage_current_log.csv"

run_ids = os.listdir(os.path.join(current_dir, dir, subdir))


def plot_arm_voltages(df: pd.DataFrame, title: str, ax: plt.Axes):
    voltages = df.filter(regex="actuator_[1-7]_voltage")
    x = np.arange(len(df))
    for i in range(7):
        ax.plot(x, voltages[f"actuator_{i+1}_voltage"], label=f"actuator_{i+1}_voltage")
    ax.set_title(title)
    ax.legend()


def plot_base_voltages(df: pd.DataFrame, title: str, ax: plt.Axes):
    voltages = df.filter(regex="bus_voltage_[1-4]")
    x = np.arange(len(df))
    for i in range(4):
        ax.plot(x, voltages[f"bus_voltage_{i+1}"], label=f"bus_voltage_{i+1}")
    ax.set_title(title)
    ax.legend()


def plot_voltages(fname1: str, fname2: str, fname3: str, run_id: str) -> plt.Figure:
    file_path1 = os.path.join(current_dir, dir, subdir, run_id, fname1)
    file_path2 = os.path.join(current_dir, dir, subdir, run_id, fname2)
    file_path3 = os.path.join(current_dir, dir, subdir, run_id, fname3)

    df1 = pd.read_csv(file_path1, index_col=False)
    df2 = pd.read_csv(file_path2, index_col=False)
    df3 = pd.read_csv(file_path3, index_col=False)

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    plot_arm_voltages(df1, "Kionva Left", ax[0])
    plot_arm_voltages(df2, "Kionva Right", ax[1])
    plot_base_voltages(df3, "Mobile Base", ax[2])

    plt.subplots_adjust(bottom=0.2)

    # read the readme file and add the text to the bottom of the plot
    readme_file = os.path.join(current_dir, dir, subdir, run_id, "readme.md")

    if os.path.exists(readme_file):
        with open(readme_file, "r") as f:
            readme = f.read()
        fig.text(0.05, 0.01, readme, fontsize=8, ha="left", va="bottom", wrap=True, bbox=dict(facecolor='white', alpha=1))

    return fig


# run_id format: dd_mm_yyyy_hh_mm_ss
# sort run_ids
run_ids = sorted(run_ids)

# save all runs
for run_id in run_ids:
    fig = plot_voltages(file_name1, file_name2, file_name3, run_id)
    save_path = os.path.join(current_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"voltages_plot_{run_id}.png"))
    plt.close(fig)
