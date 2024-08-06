import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from typing import Tuple

WHEEL_COORDINATES = np.array((
    [0.188, 0.2075],
    [-0.188, 0.2075],
    [-0.188, -0.2075],
    [0.188, -0.2075],
))


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
        legend: bool = True
    ):
        g = sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            estimator=None,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label if legend else None,
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

    def get_point_at_distance(self, x1, y1, x2, y2, distance):
        direction_x = x2 - x1
        direction_y = y2 - y1
        length = self.distance(x1, y1, x2, y2)
        unit_direction_x = direction_x / length
        unit_direction_y = direction_y / length
        new_x = x1 + distance * unit_direction_x
        new_y = y1 + distance * unit_direction_y
        return new_x, new_y

    def translate_point_with_odom(self, point):
        ox = self.mb_df["x_platform_x"].iloc[-1]
        oy = self.mb_df["x_platform_y"].iloc[-1]
        oqz = self.mb_df["x_platform_qz"].iloc[-1]

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
    ):
        # draw the line between the xy position of the given dataframes
        reex = self.kr_df["ee_s_x"][data_index]
        reey = self.kr_df["ee_s_y"][data_index]
        leex = self.kl_df["ee_s_x"][data_index]
        leey = self.kl_df["ee_s_y"][data_index]

        self.plot_line(ax, [reey, leey], [reex, leex], "blue", 2, "-", "Table", legend)
        self.plot_marker(ax, [reey, leey], [reex, leex], "black", "o", 100)

        rabx = self.kr_df["arm_base_s_x"][data_index]
        raby = self.kr_df["arm_base_s_y"][data_index]
        labx = self.kl_df["arm_base_s_x"][data_index]
        laby = self.kl_df["arm_base_s_y"][data_index]

        if use_odometry:
            rbase = np.array([rabx, raby, 0])
            lbase = np.array([labx, laby, 0])

            rbase = self.translate_point_with_odom(rbase)
            lbase = self.translate_point_with_odom(lbase)

            rabx, raby = rbase[:2]
            labx, laby = lbase[:2]

        self.plot_line(ax, [raby, laby], [rabx, labx], "red", 2, "-", "Base", legend)
        self.plot_marker(ax, [raby], [rabx], "black", "o", 100)

        # # find the points of the line that are at 0.75 distance from ee
        # x1_75, y1_75 = self.get_point_at_distance(reex, reey, rabx, reey, 0.6)
        # x2_75, y2_75 = self.get_point_at_distance(leex, leey, labx, leey, 0.6)

        # # plot the 0.75 distance line
        # # self.plot_line(ax, [reey, leey], [x1_75, x2_75], "red", 1, "--", "Target")

        # plot line from shoulder to elbow and elbow to end effector
        rex = self.kr_df["elbow_s_x"][data_index]
        rey = self.kr_df["elbow_s_y"][data_index]

        self.plot_line(
            ax, [raby, rey], [rabx, rex], "green", 1, "--", "Right Arm", legend
        )
        self.plot_line(ax, [rey, reey], [rex, reex], "green", 1, "--", legend=legend)
        self.plot_marker(ax, [raby], [rabx], "black", "o", 100)
        self.plot_marker(ax, [rey], [rex], "black", "o", 100)

        # if use_odometry:
        #     rex0 = self.kr_df["elbow_s_x"][0]
        #     rey0 = self.kr_df["elbow_s_y"][0]

        #     rabx0 = self.kr_df["arm_base_s_x"][0]
        #     raby0 = self.kr_df["arm_base_s_y"][0]

        #     # fill the area
        #     x = [rabx0, rabx, rex0, rex]
        #     y = [raby0, raby, rey0, rey]
        #     # ax.fill(y, x, "green", alpha=0.15, linewidth=0)

        # # plot arrow
        # # self.plot_arrow(ax, raby, rabx, reey - raby, reex - rabx, "orange")

        lex = self.kl_df["elbow_s_x"][data_index]
        ley = self.kl_df["elbow_s_y"][data_index]

        self.plot_line(
            ax, [laby, ley], [labx, lex], "gray", 1, "--", "Left Arm", legend
        )
        self.plot_line(ax, [ley, leey], [lex, leex], "gray", 1, "--", legend=legend)
        self.plot_marker(ax, [laby], [labx], "black", "o", 100)
        self.plot_marker(ax, [ley], [lex], "black", "o", 100)

        # # plot arrow
        # # self.plot_arrow(ax, laby, labx, leey - laby, leex - labx, "orange")

    def plot_uc_data(self, ax: plt.Axes, data_index: int):

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

        # plot the force vectors
        self.plot_arrow(ax, kl_aby, kl_abx, kl_f_y / 10, kl_f_x / 10, color=(0.75, 0, 0))
        self.plot_arrow(ax, kr_aby, kr_abx, kr_f_y / 10, kr_f_x / 10, color=(0, 0.75, 0))

        # add force vectors
        f_x = kl_f_x + kr_f_x
        f_y = kl_f_y + kr_f_y
        center = self.get_base_center(data_index)
        # color = blue + green
        self.plot_arrow(ax, center[1], center[0], f_y / 10, f_x / 10, color=(0.75, 0.75, 0.))


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

        self.plot_arrow(ax, center_point[1], center_point[0], fy / 10, fx / 10, "red")

    def plot_base_odometry(self, ax: plt.Axes):
        odom = self.mb_df.filter(regex="x_platform_x|x_platform_y|x_platform_qz")

        sns.lineplot(
            x=odom["x_platform_y"],
            y=odom["x_platform_x"],
            ax=ax,
            estimator=None,
            linewidth=1,
            linestyle="-",
            color="black",
            label="Odometry",
        )

        # Number of quivers to plot
        num_quivers = 20

        # Generate evenly spaced indices, excluding the first and last 250 points
        indices = np.linspace(500, len(odom) - 500 - 1, num_quivers, dtype=int)

        # Select the data points using the generated indices
        qz_values = odom["x_platform_qz"].iloc[indices]
        sin_qz = np.sin(qz_values / 2)
        cos_qz = np.cos(qz_values / 2)

        # Precompute the rotation matrices
        rotations = R.from_quat(
            np.column_stack(
                (np.zeros_like(sin_qz), np.zeros_like(sin_qz), sin_qz, cos_qz)
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
            rotated_points[:, 1],
            rotated_points[:, 0],
            color="blue",
            scale=5,
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
            [self.translate_point_with_odom(point) for point in wheel_coords_with_z]
        )

        # plot the wheel coordinates
        self.plot_base_wheel_coords(wheel_coords, ax)

    def plot_base_wheel_coords(self, wheel_coords: np.ndarray, ax: plt.Axes):
        sns.scatterplot(
            x=wheel_coords[:, 1],
            y=wheel_coords[:, 0],
            ax=ax,
            s=100,
            color="black",
            marker="o",
            label="Wheel",
        )

        # fll the rectangle
        y = [coord[0] for coord in wheel_coords]
        x = [coord[1] for coord in wheel_coords]
        ax.fill(x, y, "gray", alpha=0.15, linewidth=0)

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
