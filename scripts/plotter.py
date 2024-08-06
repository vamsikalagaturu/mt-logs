import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from typing import Tuple

WHEEL_COORDINATES = [
    [0.188, 0.2075],
    [-0.188, 0.2075],
    [-0.188, -0.2075],
    [0.188, -0.2075],
]


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
        self.uc_file = "uc_log.csv"

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

    def load_uc_data(self, run_id: str):
        self.run_id = run_id
        uc_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.uc_file
        )
        self.uc_df = pd.read_csv(uc_file_path, index_col=False)

    def load_data(self, run_id: str):
        self.load_kr_data(run_id)
        self.load_kl_data(run_id)
        self.load_mb_data(run_id)
        self.load_uc_data(run_id)
        self.run_id = run_id

    def set_sns_props(self):
        sns.set_theme(style="darkgrid")
        # sns.set_context("talk")
        sns.set_palette("deep")

    def create_subplots(
        self, nrows: int, ncols: int, figsize: tuple
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, axs

    def plot_elbow_and_ee_z(
        self,
        arm_df: pd.DataFrame,
        elbow_ax: plt.Axes,
        ee_ax: plt.Axes,
        arm: str = "right",
    ):
        # get the data
        elbow_z = arm_df[["elbow_s_z"]]
        ee_z = arm_df[["ee_s_z"]]
        x = np.arange(len(arm_df)) / 1000

        sns.lineplot(x=x, y=elbow_z["elbow_s_z"], label=f"{arm}_elbow_s_z", ax=elbow_ax)
        elbow_ax.set_xlabel("Time (s)")
        elbow_ax.set_ylabel("Position (m)")
        elbow_ax.set_title("Elbow Z Position")
        elbow_ax.legend()

        sns.lineplot(x=x, y=ee_z["ee_s_z"], label="ee_s_z", ax=ee_ax)
        ee_ax.set_xlabel("Time (s)")
        ee_ax.set_ylabel("Position (m)")
        ee_ax.set_title("End Effector Z Position")
        ee_ax.legend()

    def plot_elbow_z_command_force(
        self,
        arm_df: pd.DataFrame,
        ax: plt.Axes,
        arm: str = "right",
    ):
        # get the data
        elbow_z_force = arm_df[["elbow_f_c_z"]]
        x = np.arange(len(arm_df)) / 1000

        # plot the data
        sns.lineplot(
            x=x, y=elbow_z_force["elbow_f_c_z"], label=f"{arm}_elbow_f_c_z", ax=ax
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Command Force (N)")
        ax.set_title("Elbow Z Force")
        ax.legend()

    def plot_ee_orientation(
        self,
        arm_df: pd.DataFrame,
        ax: plt.Axes,
        arm: str = "right",
    ):
        # get the data
        qw = arm_df[["ee_s_qw"]]
        x = np.arange(len(arm_df)) / 1000

        # convert the quaternion to angle of rotation
        angles = 2 * np.arccos(qw)
        angles = np.degrees(angles)

        # plot the data
        sns.lineplot(x=x, y=angles["ee_s_qw"], label=f"{arm}_ee_s_qw", ax=ax)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("End Effector Orientation")
        ax.legend()

    def plot_arm_trajectory(
        self,
        arm_df: pd.DataFrame,
        ax: plt.Axes,
        coord="xz",
        arm: str = "right",
    ):

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
            label=f"{arm}_ee_s_{coord}",
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

    def plot_line(
        self,
        ax: plt.Axes,
        x,
        y,
        color: str = "black",
        linewidth: int = 1,
        linestyle: str = "-",
        label: str = None,
    ):
        sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            estimator=None,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label,
        )

    def plot_line_over_time(
        self,
        ax: plt.Axes,
        x1s: pd.Series,
        y1s: pd.Series,
        x2s: pd.Series,
        y2s: pd.Series,
        color: str = "black",
        linewidth: int = 1,
        linestyle: str = "-",
    ):

        # flatten the data
        x1 = x1s.to_list()
        y1 = y1s.to_list()
        x2 = x2s.to_list()
        y2 = y2s.to_list()

        # plot the first and last line
        self.plot_line(ax, x1[0], y1[0], x2[0], y2[0], color, linewidth, linestyle)
        self.plot_line(ax, x1[-1], y1[-1], x2[-1], y2[-1], color, linewidth, linestyle)

        # plot the intermediate lines as a shaded area
        x = x1 + x2
        y = y1 + y2

        ax.fill(x, y, color, alpha=0.75, linewidth=0)

    def plot_ee_and_shoulder_lines_over_time(
        self,
        ax: plt.Axes,
    ):
        # draw the line between the xy position of the given dataframes
        x1 = self.kr_df["ee_s_x"]
        y1 = self.kr_df["ee_s_y"]
        x2 = self.kl_df["ee_s_x"]
        y2 = self.kl_df["ee_s_y"]

        self.plot_line_over_time(ax, y1, x1, y2, x2, "blue", 2, "-")

        x1 = self.kr_df["arm_base_s_x"]
        y1 = self.kr_df["arm_base_s_y"]
        x2 = self.kl_df["arm_base_s_x"]
        y2 = self.kl_df["arm_base_s_y"]

        self.plot_line_over_time(ax, y1, x1, y2, x2, "red", 2, "-")

        # plot line from shoulder to elbow and elbow to end effector
        rsx = self.kr_df["arm_base_s_x"]
        rsy = self.kr_df["arm_base_s_y"]
        rex = self.kr_df["elbow_s_x"]
        rey = self.kr_df["elbow_s_y"]
        reex = self.kr_df["ee_s_x"]
        reey = self.kr_df["ee_s_y"]

        self.plot_line_over_time(ax, rsy, rsx, rey, rex, "green", 1, "--")
        self.plot_line_over_time(ax, rey, rex, reey, reex, "green", 1, "--")

        lsx = self.kl_df["arm_base_s_x"]
        lsy = self.kl_df["arm_base_s_y"]
        lex = self.kl_df["elbow_s_x"]
        ley = self.kl_df["elbow_s_y"]
        leex = self.kl_df["ee_s_x"]
        leey = self.kl_df["ee_s_y"]

        self.plot_line_over_time(ax, lsy, lsx, ley, lex, "gray", 1, "--")
        self.plot_line_over_time(ax, ley, lex, leey, leex, "gray", 1, "--")

    def plot_marker(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        color: str = "black",
        marker: str = "o",
        size: int = 100,
    ):
        sns.scatterplot(
            x=x,
            y=y,
            ax=ax,
            color=color,
            s=size,
            marker=marker,
        )

    def plot_arrow(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        dx: float,
        dy: float,
        color: str = "black",
        width: int = 0.0025,
    ):
        ax.arrow(
            x,
            y,
            dx,
            dy,
            color=color,
            width=width,
            length_includes_head=True,
            head_width=0.02,
            head_length=0.02,
            linewidth=1.5,
        )

        # ax.annotate(
        #     "",
        #     xy=(x + dx, y + dy),
        #     xytext=(x, y),
        #     arrowprops=dict(
        #         arrowstyle="-|>",
        #         color=color,
        #         mutation_scale=25,
        #         connectionstyle="arc3,rad=0",
        #         fc=color,
        #         lw=3,
        #     ),
        # )

    def get_base_center(self, data_index: int):
        x1 = self.kr_df["arm_base_s_x"][data_index]
        y1 = self.kr_df["arm_base_s_y"][data_index]
        x2 = self.kl_df["arm_base_s_x"][data_index]
        y2 = self.kl_df["arm_base_s_y"][data_index]

        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def plot_ee_and_shoulder_lines(
        self,
        ax: plt.Axes,
        data_index: int,
    ):
        # draw the line between the xy position of the given dataframes
        x1 = self.kr_df["ee_s_x"][data_index]
        y1 = self.kr_df["ee_s_y"][data_index]
        x2 = self.kl_df["ee_s_x"][data_index]
        y2 = self.kl_df["ee_s_y"][data_index]

        self.plot_line(ax, [y1, y2], [x1, x2], "blue", 2, "-", "Table")
        self.plot_marker(ax, [y1, y2], [x1, x2], "black", "o", 100)

        x1b = self.kr_df["arm_base_s_x"][data_index]
        y1b = self.kr_df["arm_base_s_y"][data_index]
        x2b = self.kl_df["arm_base_s_x"][data_index]
        y2b = self.kl_df["arm_base_s_y"][data_index]

        self.plot_line(ax, [y1b, y2b], [x1b, x2b], "red", 2, "-", "Base")
        self.plot_marker(ax, [y1b], [x1b], "black", "o", 100)

        # distance between the arm shoulder to ee
        kr_dist = self.distance(x1, y1, x1b, y1b)
        kl_dist = self.distance(x2, y2, x2b, y2b)

        # find the points of the line that are at 0.75 distance from ee
        def get_point_at_distance(x1, y1, x2, y2, distance):
            direction_x = x2 - x1
            direction_y = y2 - y1
            length = self.distance(x1, y1, x2, y2)
            unit_direction_x = direction_x / length
            unit_direction_y = direction_y / length
            new_x = x1 + distance * unit_direction_x
            new_y = y1 + distance * unit_direction_y
            return new_x, new_y

        x1_75, y1_75 = get_point_at_distance(x1, y1, x1b, y1, 0.6)
        x2_75, y2_75 = get_point_at_distance(x2, y2, x2b, y2, 0.6)

        dt1 = self.distance(x1, y1, x1_75, y1_75)
        dt2 = self.distance(x2, y2, x2_75, y2_75)

        print(f"distances: {kr_dist}, {kl_dist}, {dt1}, {dt2}")

        # plot the 0.75 distance line
        self.plot_line(ax, [y1, y2], [x1_75, x2_75], "blue", 1, "--", "Target")
        
        # plot line from shoulder to elbow and elbow to end effector
        rsx = self.kr_df["arm_base_s_x"][data_index]
        rsy = self.kr_df["arm_base_s_y"][data_index]
        rex = self.kr_df["elbow_s_x"][data_index]
        rey = self.kr_df["elbow_s_y"][data_index]
        reex = self.kr_df["ee_s_x"][data_index]
        reey = self.kr_df["ee_s_y"][data_index]

        self.plot_line(ax, [rsy, rey], [rsx, rex], "green", 1, "--", "Right Arm")
        self.plot_line(ax, [rey, reey], [rex, reex], "green", 1, "--")
        self.plot_marker(ax, [rsy], [rsx], "black", "o", 100)
        self.plot_marker(ax, [rey], [rex], "black", "o", 100)

        # plot arrow
        self.plot_arrow(ax, rsy, rsx, reey - rsy, reex - rsx, "orange")

        lsx = self.kl_df["arm_base_s_x"][data_index]
        lsy = self.kl_df["arm_base_s_y"][data_index]
        lex = self.kl_df["elbow_s_x"][data_index]
        ley = self.kl_df["elbow_s_y"][data_index]
        leex = self.kl_df["ee_s_x"][data_index]
        leey = self.kl_df["ee_s_y"][data_index]

        self.plot_line(ax, [lsy, ley], [lsx, lex], "gray", 1, "--", "Left Arm")
        self.plot_line(ax, [ley, leey], [lex, leex], "gray", 1, "--")
        self.plot_marker(ax, [lsy], [lsx], "black", "o", 100)
        self.plot_marker(ax, [ley], [lex], "black", "o", 100)

        # plot arrow
        self.plot_arrow(ax, lsy, lsx, leey - lsy, leex - lsx, "orange")

    def plot_uc_data(self, ax: plt.Axes, data_index: int):
        kl_bl_base_dist = self.uc_df["kl_bl_base_dist"][data_index]
        kr_br_base_dist = self.uc_df["kr_bl_base_dist"][data_index]

        print(f"dists: {kl_bl_base_dist}, {kr_br_base_dist}")

    def plot_base_force_direction(
        self, ax: plt.Axes, data_index: int, center_point: list
    ):
        # get the data
        fx = self.mb_df["platform_force_x"][data_index]
        fy = self.mb_df["platform_force_y"][data_index]
        mz = self.mb_df["platform_force_z"][data_index]

        print(f"force: {fx}, {fy}, {mz}")
        
        norm = np.linalg.norm([fx, fy])
        fx /= norm
        fy /= norm

        # plot the force vector
        # ax.quiver(
        #     center_point[1],
        #     center_point[0],
        #     -fy,
        #     fx,
        #     color="red",
        #     scale=10,
        #     scale_units="xy",
        #     width=0.005,
        # )

        self.plot_arrow(ax, center_point[1], center_point[0], fy/10, fx/10, "red")

    def plot_base_odometry(self, ax: plt.Axes):
        odom = self.mb_df.filter(regex="x_platform_x|x_platform_y|x_platform_qz")

        # plot the data
        sns.lineplot(y=odom["x_platform_x"], x=odom["x_platform_y"], ax=ax)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")

    def plot_base_wheel_coords(self, ax: plt.Axes):
        for coord in WHEEL_COORDINATES:
            sns.scatterplot(
                y=[coord[0]],
                x=[coord[1]],
                ax=ax,
                color="red",
                s=100,
                marker="o",
            )

        # fll the rectangle
        y = [coord[0] for coord in WHEEL_COORDINATES]
        x = [coord[1] for coord in WHEEL_COORDINATES]
        ax.fill(x, y, "gray", alpha=0.25, linewidth=0)

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
