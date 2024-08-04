import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
import os
from typing import Annotated

current_dir = os.path.dirname(__file__)

dir = "data"
subdir = "freddy_base_control"
save_dir = f"plots/{subdir}"
file_name1 = "kinova_left_log.csv"
file_name2 = "kinova_right_log.csv"
file_name3 = "mobile_base_log.csv"

run_ids = os.listdir(os.path.join(current_dir, dir, subdir))

wheel_coordinates = [
    [0.188, 0.2075],
    [-0.188, 0.2075],
    [-0.188, -0.2075],
    [0.188, -0.2075],
]

def plot_pid(axes: np.ndarray, df: pd.DataFrame):
    # plot pid controller data
    p = df[f"p"]
    i = df[f"i"]
    d = df[f"d"]
    error_sum = df[f"error_sum"]
    measured = df[f"measured_value"]
    reference = df[f"reference_value"]
    error = df[f"error"]

    # plot error 
    axes[0].plot(error, label="error")
    axes[0].set_title("Error")

    # plot error sum
    axes[1].plot(error_sum, label="error_sum")
    axes[1].set_title("Error Sum")

    # plot p, i, d in the same plot
    x = range(len(p))
    axes[2].plot(x, p, label="p")
    axes[2].plot(x, i, label="i")
    axes[2].plot(x, d, label="d")
    axes[2].set_title("PID Values")

    # plot measured and reference values
    axes[3].plot(measured, label="measured")
    axes[3].plot(reference, label="reference")
    axes[3].set_title("Measured vs Reference")


def plot_base(ax: plt.Axes):
    # plot wheel coordinates by joining lines
    ax.plot(
        [wheel_coordinates[0][1], wheel_coordinates[1][1]],
        [wheel_coordinates[0][0], wheel_coordinates[1][0]],
        color="red",
    )
    ax.plot(
        [wheel_coordinates[1][1], wheel_coordinates[2][1]],
        [wheel_coordinates[1][0], wheel_coordinates[2][0]],
        color="red",
    )
    ax.plot(
        [wheel_coordinates[2][1], wheel_coordinates[3][1]],
        [wheel_coordinates[2][0], wheel_coordinates[3][0]],
        color="red",
    )
    ax.plot(
        [wheel_coordinates[3][1], wheel_coordinates[0][1]],
        [wheel_coordinates[3][0], wheel_coordinates[0][0]],
        color="red",
    )


def plot_link_point(ax: plt.Axes, df: pd.DataFrame, name: str, color: str):
    # get x and y coordinates of the link point
    # format: name_s_x, name_s_y
    x = df[f"{name}_s_x"][0]
    y = df[f"{name}_s_y"][0]

    # plot the link point
    ax.scatter(y, x, label=name, color=color)


def plot_arm(df: pd.DataFrame, title: str, ax: plt.Axes):
    # plot wheel coordinates
    plot_base(ax)

    # plot ee
    plot_link_point(ax, df, "ee", "blue")

    # plot elbow
    plot_link_point(ax, df, "elbow", "green")

    # plot arm base
    plot_link_point(ax, df, "arm_base", "red")

    # aspect ratio
    ax.set_aspect("equal")

    # limit x and y axis
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    # invert x axis
    ax.invert_xaxis()

    ax.set_title(title)


def plot_odom(df: pd.DataFrame, title: str, ax: plt.Axes):
    odom = df.filter(regex="x_platform_x|x_platform_y|x_platform_qz")
    # plot x vs y
    ax.plot(odom["x_platform_x"], odom["x_platform_y"], label="x vs y")
    ax.set_title(title)


def plot_data(fname1: str, fname2: str, fname3: str, run_id: str) -> plt.Figure:
    file_path1 = os.path.join(current_dir, dir, subdir, run_id, fname1)
    file_path2 = os.path.join(current_dir, dir, subdir, run_id, fname2)
    file_path3 = os.path.join(current_dir, dir, subdir, run_id, fname3)

    df1 = pd.read_csv(file_path1, index_col=False)
    df2 = pd.read_csv(file_path2, index_col=False)
    df3 = pd.read_csv(file_path3, index_col=False)

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    plot_arm(df1, "Kionva Left", ax[0])
    plot_arm(df2, "Kionva Right", ax[1])
    plot_odom(df3, "Mobile Base odom", ax[2])

    plt.subplots_adjust(bottom=0.2)

    # read the readme file and add the text to the bottom of the plot
    readme_file = os.path.join(current_dir, dir, subdir, run_id, "run_description.md")

    if os.path.exists(readme_file):
        with open(readme_file, "r") as f:
            readme = f.read()
        fig.text(
            0.05,
            0.01,
            readme,
            fontsize=8,
            ha="left",
            va="bottom",
            wrap=True,
            bbox=dict(facecolor="white", alpha=1),
        )

    return fig

def plot_pid_data(fname: str, run_id: str) -> plt.Figure:
    file_path = os.path.join(current_dir, dir, subdir, run_id, fname)
    df = pd.read_csv(file_path, index_col=False)

    fig, ax = plt.subplots(4, 1, figsize=(10, 14))
    plot_pid(ax, df)

    plt.tight_layout()

    return fig

# run_id format: dd_mm_yyyy_hh_mm_ss
# sort run_ids
run_ids = sorted(run_ids)

# save all runs
for run_id in run_ids:
    # fig = plot_data(file_name1, file_name2, file_name3, run_id)

    # get control data file names starting with "control"
    control_files = [f for f in os.listdir(os.path.join(current_dir, dir, subdir, run_id)) if f.startswith("control")]

    for control_file in control_files:
        fig = plot_pid_data(control_file, run_id)
        save_path = os.path.join(current_dir, save_dir, run_id)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"uc1_{control_file}.png"))
        plt.close(fig)

    # save_path = os.path.join(current_dir, save_dir)
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, f"uc1_{run_id}.png"))
    # plt.close(fig)
