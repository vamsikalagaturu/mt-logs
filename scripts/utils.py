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

    def load_data(self, run_id: str):
        self.run_id = run_id
        kr_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.kr_file
        )
        # kl_file_path = os.path.join(
        #     self.current_dir, self.data_dir, self.run_dir, run_id, self.kl_file
        # )
        # mb_file_path = os.path.join(
        #     self.current_dir, self.data_dir, self.run_dir, run_id, self.mb_file
        # )

        self.kr_df = pd.read_csv(kr_file_path, index_col=False)
        # self.kl_df = pd.read_csv(kl_file_path, index_col=False)
        # self.mb_df = pd.read_csv(mb_file_path, index_col=False)

    def set_sns_props(self):
        sns.set_theme(style="darkgrid")
        # sns.set_context("talk")
        sns.set_palette("deep")

    def plot_ee_orientation(self, arm_df: pd.DataFrame, arm: str, show: bool = False):
        # get the data
        qw = arm_df[["ee_s_qw"]]
        x = np.arange(len(arm_df))

        # convert the quaternion to angle of rotation 
        angles = 2 * np.arccos(qw)
        angles = np.degrees(angles)

        # plot the data
        sns.lineplot(x=x, y=angles["ee_s_qw"], label="ee_s_qw")

        plt.xlabel("Time")
        plt.ylabel("Angle (degrees)")
        plt.title("End Effector Orientation")
        plt.legend()
        
        if show:
            plt.show()

    def save_fig(self, file_name: str):
        assert file_name is not None, "file_name cannot be None"

        save_path = os.path.join(self.current_dir, self.save_dir, self.run_id)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{file_name}.png"))
