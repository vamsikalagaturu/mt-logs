import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from typing import Annotated


class Plotter:
    def __init__(self, run_dir: str) -> None:
        # set the output path to be the root of the project
        self.current_dir = os.path.dirname(__file__)
        self.data_dir = "../data"
        self.run_dir = run_dir
        self.save_dir = f"../plots/{run_dir}"

        self.kr_file = "kinova_right_log.csv"
        self.kl_file = "kinova_left_log.csv"
        self.mb_file = "mobile_base_log.csv"

        self.run_id = None

        self.set_sns_props()

    def load_kr_data(self, run_id: str):
        self.run_id = run_id
        kr_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.kr_file
        )
        self.kr_df = pd.read_csv(kr_file_path, index_col=False)

    def load_kl_data(self, run_id: str):
        self.run_id = run_id
        kl_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.kl_file
        )
        self.kl_df = pd.read_csv(kl_file_path, index_col=False)

    def load_mb_data(self, run_id: str):
        self.run_id = run_id
        mb_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.mb_file
        )
        self.mb_df = pd.read_csv(mb_file_path, index_col=False)

    def load_data(self, run_id: str):
        self.run_id = run_id
        self.load_kr_data(run_id)
        self.load_kl_data(run_id)
        self.load_mb_data(run_id)

    def set_sns_props(self):
        sns.set_theme(style="darkgrid")
        # sns.set_context("talk")
        sns.set_palette("deep")

    def create_subplots(self, nrows: int, ncols: int, figsize: tuple):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, axs

    def plot_elbow_and_ee_z(
        self, arm_df: pd.DataFrame, elbow_ax: plt.Axes, ee_ax: plt.Axes
    ):
        # get the data
        elbow_z = arm_df[["elbow_s_z"]]
        ee_z = arm_df[["ee_s_z"]]
        x = np.arange(len(arm_df)) / 1000

        sns.lineplot(x=x, y=elbow_z["elbow_s_z"], label="elbow_s_z", ax=elbow_ax)
        elbow_ax.set_xlabel("Time (s)")
        elbow_ax.set_ylabel("Position (m)")
        elbow_ax.set_title("Elbow Z Position")
        elbow_ax.legend()

        sns.lineplot(x=x, y=ee_z["ee_s_z"], label="ee_s_z", ax=ee_ax)
        ee_ax.set_xlabel("Time (s)")
        ee_ax.set_ylabel("Position (m)")
        ee_ax.set_title("End Effector Z Position")
        ee_ax.legend()

    def plot_elbow_z_command_force(self, arm_df: pd.DataFrame, ax: plt.Axes):
        # get the data
        elbow_z_force = arm_df[["elbow_f_c_z"]]
        x = np.arange(len(arm_df)) / 1000

        # plot the data
        sns.lineplot(x=x, y=elbow_z_force["elbow_f_c_z"], label="elbow_f_c_z", ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Command Force (N)")
        ax.set_title("Elbow Z Force")
        ax.legend()

    def plot_ee_orientation(self, arm_df: pd.DataFrame, ax: plt.Axes):
        # get the data
        qw = arm_df[["ee_s_qw"]]
        x = np.arange(len(arm_df)) / 1000

        # convert the quaternion to angle of rotation
        angles = 2 * np.arccos(qw)
        angles = np.degrees(angles)

        # plot the data
        sns.lineplot(x=x, y=angles["ee_s_qw"], label="ee_s_qw", ax=ax)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("End Effector Orientation")
        ax.legend()

    def plot_arm_trajectory(self, arm_df: pd.DataFrame, ax: plt.Axes, coord="xz"):

        coord_data = {
            "xy": ["ee_s_x", "ee_s_y"],
            "xz": ["ee_s_x", "ee_s_z"],
            "yz": ["ee_s_y", "ee_s_z"],
        }

        # plot xz trajectory of the end effector
        ee_coord = arm_df[coord_data[coord]]
        t = np.arange(len(arm_df)) / 1000

        ax.plot(
            ee_coord[coord_data[coord][0]],
            ee_coord[coord_data[coord][1]],
            label=f"ee_s_{coord}",
            linewidth=1,
        )
        # sns.regplot(
        #     x=ee_xz["ee_s_x"],
        #     y=ee_xz["ee_s_z"],
        #     ax=ax,
        #     fit_reg=False,
        #     scatter_kws={"s": 1},
        #     line_kws={"color": "red"},
        # )
        ax.set_xlabel(f'{coord_data[coord][0].split("_")[-1].upper()} Position (m)')
        ax.set_ylabel(f'{coord_data[coord][1].split("_")[-1].upper()} Position (m)')
        ax.set_title("End Effector Trajectory")
        ax.legend()

    def plot_3d_arm_trajectory(self, arm_df: pd.DataFrame):

        # get the data
        ee_xyz = arm_df[["ee_s_x", "ee_s_y", "ee_s_z"]]
        ee_xyz = ee_xyz.to_numpy()

        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2])

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("End Effector Trajectory")

        plt.show()

    def save_fig(self, file_name: str):
        assert file_name is not None, "file_name cannot be None"

        # copy the readme.md file to the save directory
        readme_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, "readme.md"
        )
        save_path = os.path.join(self.current_dir, self.save_dir, self.run_id)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{file_name}.png"))

        # copy the readme.md file to the save directory
        import shutil

        shutil.copy2(readme_path, save_path)
