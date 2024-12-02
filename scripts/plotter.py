import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from typing import Tuple

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

from utils import make_legend_arrow

rc("grid", c="0.0", ls=":", lw=0.6)
rc("xtick", top=True, bottom=True, direction="in")
rc("ytick", left=True, right=True, direction="in")
rc("axes", linewidth=1.5)
rc("xtick.major", pad=7.0)
rc("ytick.major", pad=7.0)


def math_formatter(x, pos):
    return "%i" % x


WHEEL_COORDINATES = np.array(
    (
        [0.188, 0.2075],
        [-0.188, 0.2075],
        [-0.188, -0.2075],
        [0.188, -0.2075],
    )
)

gcolors = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "gray": "#999999",
    "red": "#e41a1c",
    "yellow": "#dede00",
}


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

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": "Helvetica",
                "font.size": 12,
                "text.latex.preamble": [
                    r"\usepackage{helvet}",
                    r"\usepackage{sansmath}",
                    r"\sansmath",
                ],
            }
        )

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
        sns.set_theme(style="whitegrid")
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
        prefix: str = "",
    ):
        # get the data
        elbow_z = arm_df[["elbow_s_z"]]
        ee_z = arm_df[["ee_s_z"]]
        x = np.arange(len(arm_df)) / 1000

        prefix1, prefix2 = prefix.split(",") if prefix else (None, None)

        prefix1 = f"({prefix1}) " if prefix1 else ""
        prefix2 = f"({prefix2}) " if prefix2 else ""

        sns.lineplot(x=x, y=elbow_z["elbow_s_z"], label=f"{arm}_elbow_s_z", ax=elbow_ax)
        elbow_ax.set_xlabel("Time (s)")
        elbow_ax.set_ylabel("Position (m)")
        elbow_ax.set_title(f"{prefix1}Elbow Z Position")
        elbow_ax.legend()

        sns.lineplot(x=x, y=ee_z["ee_s_z"], label=f"{arm}_ee_s_z", ax=ee_ax)
        ee_ax.set_xlabel("Time (s)")
        ee_ax.set_ylabel("Position (m)")
        ee_ax.set_title(f"{prefix2}End Effector Z Position")
        ee_ax.legend()

    def plot_ee_z(self, arm_df: pd.DataFrame, ax: plt.Axes, arm: str = "right"):
        # get the data
        ee_z = arm_df[["ee_s_z"]]
        x = np.arange(len(arm_df)) / 1000

        # plot the data
        sns.lineplot(x=x, y=ee_z["ee_s_z"], label=f"{arm}_ee_s_z", ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.legend()

    def plot_elbow_z(self, arm_df: pd.DataFrame, ax: plt.Axes, arm: str = "right"):
        # get the data
        elbow_z = arm_df[["elbow_s_z"]]
        x = np.arange(len(arm_df)) / 1000

        # plot the data
        sns.lineplot(x=x, y=elbow_z["elbow_s_z"], label=f"{arm}_elbow_s_z", ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.legend()

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
        prefix: str = "",
    ):
        # get the data
        qw = arm_df[["ee_s_qw"]]
        x = np.arange(len(arm_df)) / 1000

        # convert the quaternion to angle of rotation
        angles = 2 * np.arccos(qw)
        angles = np.degrees(angles)

        # plot the data
        sns.lineplot(x=x, y=angles["ee_s_qw"], label=f"2*arccos({arm}_ee_s_qw)", ax=ax)

        prefix = f"({prefix}) " if prefix else ""
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"{prefix}End Effector Change in Orientation")
        ax.legend()

    def plot_arm_trajectory(
        self,
        arm_df: pd.DataFrame,
        ax: plt.Axes,
        coord="xz",
        arm: str = "right",
        prefix: str = "",
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
        prefix = f"({prefix}) " if prefix else ""
        ax.set_xlabel(f'{coord_data[coord][0].split("_")[-1].upper()} Position (m)')
        ax.set_ylabel(f'{coord_data[coord][1].split("_")[-1].upper()} Position (m)')
        ax.set_title(f"{prefix}End Effector Trajectory")
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
        legend: bool = True,
    ):
        g = sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            estimator=None,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label if legend else "",
            legend=False,
        )

        if legend:
            handles, labels = ax.get_legend_handles_labels()

            # Update the legend with the combined handles and labels
            ax.legend(
                handles,
                labels,
                handler_map={
                    mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
                },
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
        self.plot_line(ax, [x1[0], x2[0]], [y1[0], y2[0]], color, linewidth, linestyle)
        self.plot_line(
            ax, [x1[-1], x2[-1]], [y1[-1], y2[-1]], color, linewidth, linestyle
        )

        # plot the intermediate lines as a shaded area
        x = x1 + x2
        y = y1 + y2

        ax.fill(x, y, color, alpha=0.25, linewidth=0)

    def plot_ee_and_shoulder_lines_over_time(
        self,
        ax: plt.Axes,
    ):
        # draw the line between the xy position of the given dataframes
        x1 = self.kr_df["ee_s_x"][:3000]
        y1 = self.kr_df["ee_s_y"][:3000]
        x2 = self.kl_df["ee_s_x"][:3000]
        y2 = self.kl_df["ee_s_y"][:3000]

        self.plot_line_over_time(ax, y1, x1, y2, x2, "blue", 2, "-")

        x1 = self.kr_df["arm_base_s_x"][:3000]
        y1 = self.kr_df["arm_base_s_y"][:3000]
        x2 = self.kl_df["arm_base_s_x"][:3000]
        y2 = self.kl_df["arm_base_s_y"][:3000]

        self.plot_line_over_time(ax, y1, x1, y2, x2, "red", 2, "-")

        # plot line from shoulder to elbow and elbow to end effector
        rsx = self.kr_df["arm_base_s_x"][:3000]
        rsy = self.kr_df["arm_base_s_y"][:3000]
        rex = self.kr_df["elbow_s_x"][:3000]
        rey = self.kr_df["elbow_s_y"][:3000]
        reex = self.kr_df["ee_s_x"][:3000]
        reey = self.kr_df["ee_s_y"][:3000]

        self.plot_line_over_time(ax, rsy, rsx, rey, rex, "green", 1, "--")
        self.plot_line_over_time(ax, rey, rex, reey, reex, "green", 1, "--")

        lsx = self.kl_df["arm_base_s_x"][:3000]
        lsy = self.kl_df["arm_base_s_y"][:3000]
        lex = self.kl_df["elbow_s_x"][:3000]
        ley = self.kl_df["elbow_s_y"][:3000]
        leex = self.kl_df["ee_s_x"][:3000]
        leey = self.kl_df["ee_s_y"][:3000]

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
            legend=False,
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
        label: str = None,
        update_legend: bool = True,
    ):
        arrow = ax.arrow(
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
            label=label,
        )

        if not update_legend:
            return

        # Retrieve existing legend handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Update the legend with the combined handles and labels
        ax.legend(
            handles,
            labels,
            handler_map={
                mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
            },
        )

    def get_base_shoulder_center(self, data_index: int):
        x1 = self.kr_df["arm_base_s_x"][data_index]
        y1 = self.kr_df["arm_base_s_y"][data_index]
        x2 = self.kl_df["arm_base_s_x"][data_index]
        y2 = self.kl_df["arm_base_s_y"][data_index]

        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def get_base_center(self, wheel_coordinates):
        return np.mean(wheel_coordinates, axis=0)

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_point_at_distance(self, x1, y1, x2, y2, distance):
        direction_x = x2 - x1
        direction_y = y2 - y1
        length = self.distance(x1, y1, x2, y2)
        unit_direction_x = direction_x / length
        unit_direction_y = direction_y / length
        new_x = x1 + distance * unit_direction_x
        new_y = y1 + distance * unit_direction_y
        return new_x, new_y

    def translate_point_with_odom(self, point, data_index: int):
        ox = self.mb_df["x_platform_x"].iloc[data_index]
        oy = self.mb_df["x_platform_y"].iloc[data_index]
        oqz = self.mb_df["x_platform_qz"].iloc[data_index]

        r = R.from_quat([0, 0, np.sin(oqz / 2), np.cos(oqz / 2)])
        point = np.array([point[0], point[1], 0])
        point = r.apply(point)
        point[0] += ox
        point[1] += oy
        return point

    def plot_ee_and_shoulder_lines(
        self,
        ax: plt.Axes,
        data_index: int,
        use_odometry: bool = False,
        legend: bool = True,
        colors: dict = None,
        linewidths: dict = None,
        only_shoulders: bool = False,
    ):
        # draw the line between the xy position of the given dataframes
        reex = self.kr_df["ee_s_x"][data_index]
        reey = self.kr_df["ee_s_y"][data_index]
        leex = self.kl_df["ee_s_x"][data_index]
        leey = self.kl_df["ee_s_y"][data_index]

        if use_odometry:
            ree = np.array([reex, reey, 0])
            lee = np.array([leex, leey, 0])

            ree = self.translate_point_with_odom(ree, data_index)
            lee = self.translate_point_with_odom(lee, data_index)

            reex, reey = ree[:2]
            leex, leey = lee[:2]

        if not only_shoulders:
            self.plot_line(
                ax,
                [reey, leey],
                [reex, leex],
                colors["table"],
                linewidths["table"],
                "-",
                "Table",
                legend,
            )
            self.plot_marker(ax, [reey, leey], [reex, leex], "darkgray", "o", 100)

        rabx = self.kr_df["arm_base_s_x"][data_index]
        raby = self.kr_df["arm_base_s_y"][data_index]
        labx = self.kl_df["arm_base_s_x"][data_index]
        laby = self.kl_df["arm_base_s_y"][data_index]

        if use_odometry:
            rbase = np.array([rabx, raby, 0])
            lbase = np.array([labx, laby, 0])

            rbase = self.translate_point_with_odom(rbase, data_index)
            lbase = self.translate_point_with_odom(lbase, data_index)

            rabx, raby = rbase[:2]
            labx, laby = lbase[:2]

        self.plot_line(
            ax,
            [raby, laby],
            [rabx, labx],
            colors["mb"],
            linewidths["mb"],
            "-",
            "Shoulders",
            legend,
        )

        if only_shoulders:
            return

        self.plot_marker(ax, [raby], [rabx], "darkgray", "o", 100)

        # print dist bw ee and shoulders
        kr_ee_s_dist = self.distance(reex, reey, rabx, raby)
        kl_ee_s_dist = self.distance(leex, leey, labx, laby)

        # print(f"kr_ee_s_dist: {kr_ee_s_dist}, kl_ee_s_dist: {kl_ee_s_dist}")

        sp_dist = self.uc_df["dist_sp"][0]

        # # find the points of the line that are at 0.75 distance from ee
        x1_75, y1_75 = self.get_point_at_distance(reex, reey, rabx, reey, sp_dist)
        x2_75, y2_75 = self.get_point_at_distance(leex, leey, labx, leey, sp_dist)

        # # plot the 0.75 distance line
        self.plot_line(ax, [reey, leey], [x1_75, x2_75], "red", 1, "--", "Target")

        # plot line from shoulder to elbow and elbow to end effector
        rex = self.kr_df["elbow_s_x"][data_index]
        rey = self.kr_df["elbow_s_y"][data_index]

        if use_odometry:
            rex, rey = self.translate_point_with_odom([rex, rey], data_index)[:2]

        self.plot_line(
            ax,
            [raby, rey],
            [rabx, rex],
            colors["kr"],
            linewidths["kr"],
            "--",
            "Right Arm",
            legend,
        )
        self.plot_line(
            ax,
            [rey, reey],
            [rex, reex],
            colors["kr"],
            linewidths["kr"],
            "--",
            legend=legend,
        )
        self.plot_marker(ax, [raby], [rabx], "darkgray", "o", 100)
        self.plot_marker(ax, [rey], [rex], "darkgray", "o", 100)

        # if use_odometry:
        #     rex0 = self.kr_df["elbow_s_x"][0]
        #     rey0 = self.kr_df["elbow_s_y"][0]

        #     rabx0 = self.kr_df["arm_base_s_x"][0]
        #     raby0 = self.kr_df["arm_base_s_y"][0]

        #     # fill the area
        #     x = [rabx0, rex0,  rex, rabx]
        #     y = [raby0, rey0,  rey, raby]
        # ax.fill(y, x, "green", alpha=0.15, linewidth=0)

        # plot arrow
        # self.plot_arrow(ax, raby, rabx, reey - raby, reex - rabx, "orange")

        lex = self.kl_df["elbow_s_x"][data_index]
        ley = self.kl_df["elbow_s_y"][data_index]

        if use_odometry:
            lex, ley = self.translate_point_with_odom([lex, ley], data_index)[:2]

        self.plot_line(
            ax,
            [laby, ley],
            [labx, lex],
            colors["kl"],
            linewidths["kl"],
            "--",
            "Left Arm",
            legend,
        )
        self.plot_line(
            ax,
            [ley, leey],
            [lex, leex],
            colors["kl"],
            linewidths["kl"],
            "--",
            legend=legend,
        )
        self.plot_marker(ax, [laby], [labx], "darkgray", "o", 100)
        self.plot_marker(ax, [ley], [lex], "darkgray", "o", 100)

        # plot arrow
        # self.plot_arrow(ax, laby, labx, leey - laby, leex - labx, "orange")

    def plot_uc_data(
        self,
        ax: plt.Axes,
        data_index: int,
        colors: dict = None,
        use_odometry: bool = False,
        legend: bool = True,
    ):
        kl_f_x = self.uc_df["kl_bl_base_f_at_base_x"][data_index]
        kl_f_y = self.uc_df["kl_bl_base_f_at_base_y"][data_index]
        kr_f_x = self.uc_df["kr_bl_base_f_at_base_x"][data_index]
        kr_f_y = self.uc_df["kr_bl_base_f_at_base_y"][data_index]

        # normalize the force vectors
        norm = np.linalg.norm([kl_f_x, kl_f_y])
        kl_f_x /= norm
        kl_f_y /= norm

        norm = np.linalg.norm([kr_f_x, kr_f_y])
        kr_f_x /= norm
        kr_f_y /= norm

        kl_abx = self.kl_df["arm_base_s_x"][data_index]
        kl_aby = self.kl_df["arm_base_s_y"][data_index]
        kr_abx = self.kr_df["arm_base_s_x"][data_index]
        kr_aby = self.kr_df["arm_base_s_y"][data_index]

        if use_odometry:
            kl_ab = np.array([kl_abx, kl_aby, 0])
            kr_ab = np.array([kr_abx, kr_aby, 0])

            kl_ab = self.translate_point_with_odom(kl_ab, data_index)
            kr_ab = self.translate_point_with_odom(kr_ab, data_index)

            kl_abx, kl_aby = kl_ab[:2]
            kr_abx, kr_aby = kr_ab[:2]

        # plot the force vectors
        self.plot_arrow(
            ax,
            kl_aby,
            kl_abx,
            kl_f_y / 10,
            kl_f_x / 10,
            color=colors["kl_f"],
            label="Left Arm Force Vector" if legend else None,
            update_legend=legend,
        )
        self.plot_arrow(
            ax,
            kr_aby,
            kr_abx,
            kr_f_y / 10,
            kr_f_x / 10,
            color=colors["kr_f"],
            label="Right Arm Force Vector" if legend else None,
            update_legend=legend,
        )

    def plot_uc2_data(
        self,
        ax: plt.Axes,
        data_index: int,
        colors: dict = None,
        use_odometry: bool = False,
    ):
        kl_f_x = -self.uc_df["kl_bl_base_f_dir_z"][data_index]
        kl_f_y = self.uc_df["kl_bl_base_f_dir_x"][data_index]
        kr_f_x = -self.uc_df["kr_bl_base_f_dir_z"][data_index]
        kr_f_y = -self.uc_df["kl_bl_base_f_dir_x"][data_index]

        # normalize the force vectors
        norm = np.linalg.norm([kl_f_x, kl_f_y])
        kl_f_x /= norm
        kl_f_y /= norm

        norm = np.linalg.norm([kr_f_x, kr_f_y])
        kr_f_x /= norm
        kr_f_y /= norm

        kl_abx = self.kl_df["ee_s_x"][data_index]
        kl_aby = self.kl_df["ee_s_y"][data_index]
        kr_abx = self.kr_df["ee_s_x"][data_index]
        kr_aby = self.kr_df["ee_s_y"][data_index]

        if use_odometry:
            kl_ab = np.array([kl_abx, kl_aby, 0])
            kr_ab = np.array([kr_abx, kr_aby, 0])

            kl_ab = self.translate_point_with_odom(kl_ab, data_index)
            kr_ab = self.translate_point_with_odom(kr_ab, data_index)

            kl_abx, kl_aby = kl_ab[:2]
            kr_abx, kr_aby = kr_ab[:2]

        # plot the force vectors
        self.plot_arrow(
            ax,
            kl_aby,
            kl_abx,
            kl_f_y / 10,
            kl_f_x / 10,
            color=colors["kl_f"],
            label="Left Arm Force Vector",
        )
        self.plot_arrow(
            ax,
            kr_aby,
            kr_abx,
            kr_f_y / 10,
            kr_f_x / 10,
            color=colors["kr_f"],
            label="Right Arm Force Vector",
        )

        kl_f_bx = self.uc_df["kl_bl_base_f_at_base_x"][data_index]
        kl_f_by = self.uc_df["kl_bl_base_f_at_base_y"][data_index]
        kr_f_bx = self.uc_df["kr_bl_base_f_at_base_x"][data_index]
        kr_f_by = self.uc_df["kr_bl_base_f_at_base_y"][data_index]

        # Standardization for kl
        mean_kl = np.mean([kl_f_bx, kl_f_by])
        std_kl = np.std([kl_f_bx, kl_f_by])
        kl_f_bx = (kl_f_bx - mean_kl) / std_kl
        kl_f_by = (kl_f_by - mean_kl) / std_kl

        # Standardization for kr
        mean_kr = np.mean([kr_f_bx, kr_f_by])
        std_kr = np.std([kr_f_bx, kr_f_by])
        kr_f_bx = (kr_f_bx - mean_kr) / std_kr
        kr_f_by = (kr_f_by - mean_kr) / std_kr

        # add force vectors
        f_x = kl_f_bx + kr_f_bx
        f_y = kl_f_by + kr_f_by

        center = self.get_base_shoulder_center(data_index)

        if use_odometry:
            center = self.translate_point_with_odom(center, data_index)

        # color = blue + green
        self.plot_arrow(
            ax,
            center[1],
            center[0],
            f_y / 10,
            f_x / 10,
            color=colors["base_f"],
            label="Resulting Force Vector at Base",
        )

    def plot_base_force_direction(
        self,
        ax: plt.Axes,
        data_index: int,
        center_point: list,
        colors: dict,
        legend: bool = True,
        final=False,
    ):
        # mark the center
        sns.scatterplot(
            x=[center_point[1]],
            y=[center_point[0]],
            ax=ax,
            s=200,
            color=colors["base_center"],
            marker="o",
            linewidth=1,
            edgecolor="black",
            label="Base Center" if legend else "",
        )

        # get the data
        fx = self.mb_df["platform_force_x"][data_index]
        fy = self.mb_df["platform_force_y"][data_index]
        mz = self.mb_df["platform_force_z"][data_index]

        # print(f"force: {fx}, {fy}, {mz}")

        norm = np.linalg.norm([fx, fy])
        fx /= norm
        fy /= norm

        if not final:
            self.plot_arrow(
                ax,
                center_point[1],
                center_point[0],
                fy / 10,
                fx / 10,
                color=colors["base_f"],
                label="Base Force Vector" if legend else "",
            )

    def plot_base_force(self, ax: plt.Axes, data_index: int, colors: dict):
        # get the data
        fx = self.mb_df["platform_force_x"]
        fy = self.mb_df["platform_force_y"]
        mz = self.mb_df["platform_force_z"]

        # plot the data fx
        x = np.arange(len(fx)) / 1000
        ax.plot(x, fx, label="Base Force Linear X")
        ax.plot(x, fy, label="Base Force Linear Y")

        ax.legend()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.set_title("(a) Base Force", fontsize=25)

    def plot_base_force_ts(self, ax: plt.Axes, colors: dict, window_size: int = 50):
        # get the data
        fr = self.uc_df["kr_bl_base_f_mag"]
        fl = self.uc_df["kl_bl_base_f_mag"]

        # add left and right forces to get the total force
        f = fr + fl

        # Calculate the moving average (simple smoothing)
        f_smooth = f.rolling(window=window_size).mean()

        x = np.arange(len(f)) / 1000  # Time in seconds

        # Plot the original data with a transparent alpha
        # ax.plot(x, f, color="red", alpha=0.2, label=r"$|F_{base}|$")

        # plot the smoothed data
        ax.plot(
            x,
            f_smooth,
            label=r"$\mathopen|\mathbf{F_{b}}\mathclose|$",
            color=gcolors["purple"],
            linewidth=5,
        )

    def plot_base_force_wrt_world_ts(self, ax: plt.Axes, colors: dict, window_size: int = 50):

        fx = self.mb_df["platform_force_x"]
        fy = self.mb_df["platform_force_y"]
        mz = self.mb_df["platform_force_z"]

        # transform the force values to the world frame using odom
        ox = self.mb_df["x_platform_x"]
        oy = self.mb_df["x_platform_y"]
        oqz = self.mb_df["x_platform_qz"]

        f_transformed = []
        for i in range(len(fx)):
            r = R.from_quat([0, 0, np.sin(oqz[i] / 2), np.cos(oqz[i] / 2)])
            f = np.array([fx[i], fy[i], 0])
            f = r.apply(f)
            f_transformed.append(f)

        f_transformed = np.array(f_transformed)

        # lin_f_mag = np.linalg.norm(f_transformed[:, :2], axis=1)
        # lin_f_mag = pd.Series(lin_f_mag)

        # # Calculate the moving average (simple smoothing)
        # lin_f_mag_smooth = lin_f_mag.rolling(window=window_size).mean()
        
        fx_smooth = pd.Series(f_transformed[:, 0]).rolling(window=window_size).mean()
        fy_smooth = pd.Series(f_transformed[:, 1]).rolling(window=window_size).mean()
        m_z_smooth = mz.rolling(window=window_size).mean()

        x = np.arange(len(fx_smooth)) / 1000  # Time in seconds

        # ax.plot(
        #     x,
        #     lin_f_mag_smooth,
        #     label=r"$\mathopen|\mathbf{F_{b_{xy}}}\mathclose|$",
        #     color=gcolors["purple"],
        #     linewidth=5,
        # )

        # plot x and y forces
        sns.lineplot(
            x=x,
            y=fx_smooth,
            ax=ax,
            color=gcolors["purple"],
            label=r"$\mathbf{F_{b_{x}}}$",
            linewidth=5,
        )

        sns.lineplot(
            x=x,
            y=fy_smooth,
            ax=ax,
            color=gcolors["red"],
            label=r"$\mathbf{F_{b_{y}}}$",
            linewidth=5,
        )

        # plot torque on different scale on the right
        ax2 = ax.twinx()
        ax2.plot(
            x,
            m_z_smooth,
            label=r"$\mathbf{M_{z}}$",
            color=gcolors["green"],
            linewidth=5,
        )

        ax2.set_ylabel("Torque (Nm)")
        ax2.yaxis.label.set_fontsize(20)
        ax2.tick_params(axis="both", which="major", labelsize=20)

    def plot_ee_force(self, ax: plt.Axes, data_index: int, colors: dict):
        # get the data
        kr_f_mag = self.uc_df["kr_bl_base_f_mag"][200:]
        kl_f_mag = self.uc_df["kl_bl_base_f_mag"][200:]

        # plot the data
        x = np.arange(len(kr_f_mag)) / 1000
        ax.plot(x, kr_f_mag, label="Right Arm Force Magnitude")
        ax.plot(x, kl_f_mag, label="Left Arm Force Magnitude")

        ax.legend()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.set_title("(b) End Effector Force Magnitude", fontsize=25)

    def plot_ee_force_ts(self, ax: plt.Axes, colors: dict, window_size: int = 50):
        # get the data
        kr_f_mag = self.uc_df["kr_bl_base_f_mag"]
        kl_f_mag = self.uc_df["kl_bl_base_f_mag"]

        # Calculate the moving average (simple smoothing)
        kr_f_mag_smooth = kr_f_mag.rolling(window=window_size).mean()
        kl_f_mag_smooth = kl_f_mag.rolling(window=window_size).mean()

        x = np.arange(len(kr_f_mag)) / 1000  # Time in seconds

        # Plot the original data with a transparent alpha
        # ax.plot(x, kr_f_mag, color=colors.get("right_ee", "blue"), alpha=0.2)
        # ax.plot(x, kl_f_mag, color=colors.get("left_ee", "red"), alpha=0.2)

        # sns.lineplot(
        #     x=x, y=kr_f_mag, ax=ax, color=colors.get("right_ee", "blue"), alpha=0.2
        # )
        # sns.lineplot(
        #     x=x, y=kl_f_mag, ax=ax, color=colors.get("left_ee", "red"), alpha=0.2
        # )

        # plot the smoothed data
        # ax.plot(
        #     x,
        #     kr_f_mag_smooth,
        #     label=r"$|F_{right\_ee}|$",
        #     color=colors.get("right_ee", "blue"),
        # )
        # ax.plot(
        #     x,
        #     kl_f_mag_smooth,
        #     label=r"$|F_{left\_ee}|$",
        #     color=colors.get("left_ee", "red"),
        # )

        sns.lineplot(
            x=x,
            y=kr_f_mag_smooth,
            ax=ax,
            color=gcolors["blue"],
            label=r"$\mathopen|\mathbf{F_{ee}}\mathclose|_\mathbf{r}$",
            linewidth=5,
        )
        sns.lineplot(
            x=x,
            y=kl_f_mag_smooth,
            ax=ax,
            color=gcolors["pink"],
            label=r"$\mathopen|\mathbf{F_{ee}}\mathclose|_\mathbf{l}$",
            linewidth=5,
        )

        ax.legend()

    def plot_dist_ts(self, ax: plt.Axes, colors: dict, window_size: int = 50):
        # get the data
        kr_ee_s_dist = self.uc_df["kr_bl_base_dist"]
        kl_ee_s_dist = self.uc_df["kl_bl_base_dist"]

        # convert m to cm
        kr_ee_s_dist *= 100
        kl_ee_s_dist *= 100

        dist_sp = self.uc_df["dist_sp"] * 100

        x = np.arange(len(kr_ee_s_dist)) / 1000  # Time in seconds

        # Plot the original data with a transparent alpha
        sns.lineplot(
            x=x,
            y=kr_ee_s_dist,
            ax=ax,
            label=r"$\mathbf{d}_{\mathbf{r}}$",
            color=gcolors["blue"],
            linewidth=5,
        )
        sns.lineplot(
            x=x,
            y=kl_ee_s_dist,
            ax=ax,
            label=r"$\mathbf{d}_{\mathbf{l}}$",
            color=gcolors["pink"],
            linewidth=5,
        )

        # plot the setpoint
        sns.lineplot(
            x=x,
            y=dist_sp,
            ax=ax,
            label=r"$\mathbf{d}_{\mathbf{sp}}$",
            color=gcolors["green"],
            linewidth=5,
        )

    def plot_base_odometry(self, ax: plt.Axes, colors: dict, data_index: int):
        odom = self.mb_df.filter(regex="x_platform_x|x_platform_y|x_platform_qz")[
            0:data_index
        ]

        ax.plot(odom["x_platform_y"], odom["x_platform_x"], color="blue")

        # Number of quivers to plot
        num_quivers = 10

        # Generate evenly spaced indices, excluding the first and last 250 points
        indices = np.linspace(0, len(odom) - 1, num_quivers, dtype=int)

        # Select the data points using the generated indices
        qz_values = odom["x_platform_qz"].iloc[indices]
        sin_qz = np.sin(qz_values / 2)
        cos_qz = np.cos(qz_values / 2)

        # Precompute the rotation matrices
        rotations = R.from_quat(
            np.column_stack(
                (np.zeros_like(sin_qz), np.zeros_like(sin_qz), cos_qz, sin_qz)
            )
        )

        # Precompute the points to be rotated
        points = np.tile([0.1, 0, 0], (len(rotations), 1))

        # Apply the rotations
        rotated_points = rotations.apply(points)

        xn = odom["x_platform_x"].iloc[indices]
        yn = odom["x_platform_y"].iloc[indices]

        ax.quiver(
            yn,
            xn,
            -rotated_points[:, 1],
            -rotated_points[:, 0],
            color="blue",
            scale=2,
            scale_units="xy",
            width=0.0025,
            alpha=0.75,
            headlength=4,
            headaxislength=3.0,
            headwidth=4,
        )

        # translate the wheel coordinates based on the odometry last position
        wheel_coords = np.array(WHEEL_COORDINATES)
        wheel_coords_with_z = np.hstack((wheel_coords, np.zeros((4, 1))))
        wheel_coords = np.array(
            [
                self.translate_point_with_odom(point, data_index)
                for point in wheel_coords_with_z
            ]
        )

        # plot the wheel coordinates
        self.plot_base_wheel_coords(wheel_coords, ax, colors)
        # plot the pivot directions
        self.plot_wheel_pivot_directions(
            ax, len(odom) - 1, wheel_coords, colors["pivot"], legend=True, initial=False
        )
        center = self.get_base_center(wheel_coords)
        self.plot_base_force_direction(ax, data_index, center, colors, final=True)

    def plot_base_wheel_coords(
        self,
        wheel_coords: np.ndarray,
        ax: plt.Axes,
        color: dict,
        label: str = "",
        initial=False,
    ):
        sns.scatterplot(
            x=wheel_coords[:, 1],
            y=wheel_coords[:, 0],
            ax=ax,
            s=100,
            color="black",
            marker="o",
            label=label,
        )

        # fll the rectangle
        y = [coord[0] for coord in wheel_coords]
        x = [coord[1] for coord in wheel_coords]

        if not initial:
            ax.fill(x, y, color=color.get("base"), label="Base")
        else:
            # connect the dots
            for i in range(4):
                ax.plot(
                    [wheel_coords[i][1], wheel_coords[(i + 1) % 4][1]],
                    [wheel_coords[i][0], wheel_coords[(i + 1) % 4][0]],
                    color=color.get("base"),
                    linewidth=2,
                )

    def plot_wheel_pivot_directions(
        self,
        ax: plt.Axes,
        data_index: int,
        wheel_coords: np.ndarray,
        color="blue",
        legend: bool = True,
        initial=False,
    ):
        pivot_1234 = self.mb_df.filter(
            regex="pivot_1|pivot_2|pivot_3|pivot_4|pivot_5"
        ).iloc[data_index]

        # pivot angle: 0 to 2pi
        pivot_1234 = pivot_1234.to_numpy()
        pivot_1234 = pivot_1234[0:4]

        # plot the pivot directions for each wheel
        for i, pivot in enumerate(pivot_1234):
            # get the wheel coordinates
            wheel = wheel_coords[i]

            if data_index == 0:
                pivot = pivot + np.pi

            # get the pivot direction
            direction = np.array([np.cos(pivot), np.sin(pivot), 0])

            label = None
            if i == 0:
                label = (
                    "Initial pivot direction" if initial else "Final pivot direction"
                )

            # plot the pivot direction
            self.plot_arrow(
                ax,
                wheel[1],
                wheel[0],
                direction[1] / 20,
                direction[0] / 20,
                color,
                label=label if legend else None,
            )

    def plot_pivot_direction(self, ax: plt.Axes, pivot: float):
        pivot_1234 = self.mb_df.filter(regex="pivot_1|pivot_2|pivot_3|pivot_4|pivot_5")[
            200:
        ]
        x = np.arange(len(pivot_1234)) / 1000
        ax.plot(x, pivot_1234["pivot_1"], label="Pivot 1")
        ax.plot(x, pivot_1234["pivot_2"], label="Pivot 2")
        ax.plot(x, pivot_1234["pivot_3"], label="Pivot 3")
        ax.plot(x, pivot_1234["pivot_4"], label="Pivot 4")

        # legend
        ax.legend()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pivot Angle (rad)")
        ax.set_title("(b) Pivot Direction", fontsize=25)

    def save_fig(self, file_name: str, title: str = None, fontsize: int = 12):
        assert file_name is not None, "file_name cannot be None"

        # copy the readme.md file to the save directory
        readme_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, self.run_id, "readme.md"
        )
        save_path = os.path.join(self.current_dir, self.save_dir, self.run_id)
        os.makedirs(save_path, exist_ok=True)

        if title is not None:
            plt.suptitle(title, fontsize=fontsize)
        else:
            # remove title
            plt.suptitle("")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"{file_name}.png"),
            format="png",
            transparent=True,
            pad_inches=0.0,
        )

        # copy the readme.md file to the save directory
        import shutil

        shutil.copy2(readme_path, save_path)
