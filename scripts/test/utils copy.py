import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
import os
from typing import Annotated

WHEEL_COORDINATES = [
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
        [WHEEL_COORDINATES[0][1], WHEEL_COORDINATES[1][1]],
        [WHEEL_COORDINATES[0][0], WHEEL_COORDINATES[1][0]],
        color="red",
    )
    ax.plot(
        [WHEEL_COORDINATES[1][1], WHEEL_COORDINATES[2][1]],
        [WHEEL_COORDINATES[1][0], WHEEL_COORDINATES[2][0]],
        color="red",
    )
    ax.plot(
        [WHEEL_COORDINATES[2][1], WHEEL_COORDINATES[3][1]],
        [WHEEL_COORDINATES[2][0], WHEEL_COORDINATES[3][0]],
        color="red",
    )
    ax.plot(
        [WHEEL_COORDINATES[3][1], WHEEL_COORDINATES[0][1]],
        [WHEEL_COORDINATES[3][0], WHEEL_COORDINATES[0][0]],
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


def plot_arm_ee_x(df: pd.DataFrame, title: str, ax: plt.Axes):
    # plot ee x
    ax.plot(df["ee_s_x"], label="ee_x")
    ax.set_title(title)


def plot_arm_ee_z(df: pd.DataFrame, title: str, ax: plt.Axes):
    # plot ee z
    ax.plot(df["ee_s_z"], label="ee_z")
    ax.set_title(title)


def plot_arm_elbow_z(df: pd.DataFrame, title: str, ax: plt.Axes):
    # plot elbow z
    ax.plot(df["elbow_s_z"], label="elbow_z")
    ax.set_title(title)


def plot_odom(df: pd.DataFrame, title: str, ax: plt.Axes):
    odom = df.filter(regex="x_platform_x|x_platform_y|x_platform_qz")
    # plot x vs y
    ax.plot(odom["x_platform_x"], odom["x_platform_y"], label="x vs y")
    ax.set_title(title)


class Plotter:
    def __init__(self, run_dir: str) -> None:
        # set the output path to be the root of the project
        self.current_dir = os.path.dirname(__file__)
        self.data_dir = "../data"
        self.run_dir = run_dir
        self.save_dir = f"../plots/{run_dir}"

        self.kr_file = "kionva_right_log.csv"
        self.kl_file = "kionva_left_log.csv"
        self.mb_file = "mobile_base_log.csv"

        self.run_id = None

        self.set_sns_props()

    def load_data(self, run_id: str):
        self.run_id = run_id
        kr_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.kr_file
        )
        kl_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.kl_file
        )
        mb_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.mb_file
        )

        self.kr_df = pd.read_csv(kr_file_path, index_col=False)
        self.kl_df = pd.read_csv(kl_file_path, index_col=False)
        self.mb_df = pd.read_csv(mb_file_path, index_col=False)

    def set_sns_props(self):
        sns.set_theme(style="darkgrid")
        # sns.set_context("talk")
        sns.set_palette("deep")

    def save_fig(self, file_name: str):
        assert file_name is not None, "file_name cannot be None"

        save_path = os.path.join(self.current_dir, self.save_dir, self.run_id)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{file_name}.png"))
