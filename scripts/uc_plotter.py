import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.ticker import FuncFormatter


def math_formatter(x, pos):
    return "%i" % x


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

    def plot_base_force_wrt_world_ts(self, ax: plt.Axes, window_size: int = 50):
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

        fx_smooth = pd.Series(f_transformed[:, 0]).rolling(window=window_size).mean()
        fy_smooth = pd.Series(f_transformed[:, 1]).rolling(window=window_size).mean()
        m_z_smooth = mz.rolling(window=window_size).mean()

        x = np.arange(len(fx_smooth)) / 1000  # Time in seconds

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

    def plot_dist_ts(self, ax: plt.Axes):
        # get the data
        kr_ee_s_dist = self.uc_df["kr_bl_base_dist"]
        kl_ee_s_dist = self.uc_df["kl_bl_base_dist"]

        # convert m to cm
        kr_ee_s_dist *= 100
        kl_ee_s_dist *= 100

        dist_sp = self.uc_df["dist_sp"] * 100

        x = np.arange(len(kr_ee_s_dist)) / 1000  # Time in seconds

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

    def plot_ee_force_ts(self, ax: plt.Axes, window_size: int = 50):
        # get the data
        kr_f_mag = self.uc_df["kr_bl_base_f_mag"]
        kl_f_mag = self.uc_df["kl_bl_base_f_mag"]

        # Calculate the moving average (simple smoothing)
        kr_f_mag_smooth = kr_f_mag.rolling(window=window_size).mean()
        kl_f_mag_smooth = kl_f_mag.rolling(window=window_size).mean()

        x = np.arange(len(kr_f_mag)) / 1000  # Time in seconds

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


class UCPlotter:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def plot_uc1_ts(self):
        # run_id = "09_08_2024_18_48_50" # going back
        # run_id = "09_08_2024_19_48_02"  # going forward
        # run_id = "09_08_2024_20_33_38"  # going side

        # run_id = "09_08_2024_16_36_02"  # going back
        # run_id = "09_08_2024_20_33_38" # going side
        # run_id = "09_08_2024_19_48_02"

        # tests
        # run_id = "07_08_2024_12_44_32"
        # run_id = "07_08_2024_13_05_08"
        # run_id = "07_08_2024_13_14_07"
        # run_id = "07_08_2024_13_18_52"
        # run_id = "07_08_2024_13_48_47" # one minute
        # run_id = "09_08_2024_19_18_59" # going forward (20s)
        run_id = "09_08_2024_19_48_02"

        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig = plt.figure(figsize=(8, 4))

        axs = fig.add_subplot(121)
        axs2 = fig.add_subplot(122)

        plotter.plot_base_force_wrt_world_ts(axs, window_size=50)
        # axs.xaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs.yaxis.set_major_formatter(FuncFormatter(math_formatter))
        axs.set_xlabel("Time [s]")
        axs.set_ylabel("Force [N]")
        # axs.xaxis.set_ticks(np.arange(0, 16, 4))
        # axs.yaxis.set_ticks(np.arange(-50, 120, 40))
        # axs.xaxis.set_ticks(np.arange(0, 3, 1))
        # axs.yaxis.set_ticks(np.arange(-120, 10, 40))
        axs.set_aspect("auto")
        axs.xaxis.label.set_fontsize(20)
        axs.yaxis.label.set_fontsize(20)
        axs.tick_params(axis="both", which="major", labelsize=20)
        axs.legend(loc="lower left", fontsize=22)

        plotter.plot_dist_ts(axs2)
        axs2.set_xlabel("Time [s]")
        axs2.set_ylabel("Distance [cm]")
        # axs2.xaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs2.yaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs2.xaxis.set_ticks(np.arange(0, 16, 4))
        # axs2.yaxis.set_ticks(np.arange(65, 100, 10))
        # axs2.xaxis.set_ticks(np.arange(0, 3, 1))
        # axs2.yaxis.set_ticks(np.arange(60, 85, 5))
        axs2.set_aspect("auto")
        axs2.xaxis.label.set_fontsize(20)
        axs2.yaxis.label.set_fontsize(20)
        axs2.tick_params(axis="both", which="major", labelsize=20)
        axs2.legend(loc="upper right", fontsize=19)

        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        plt.show()
        # plotter.save_fig("uc1_ts_backward_4")
        # plotter.save_fig("uc1_ts_backward_2_separated")
        # plotter.save_fig("uc1_ts_sideways")
        # plotter.save_fig("uc1_ts_forward")

    def plot_uc2_ts(self):
        run_id = "07_08_2024_14_42_53"  # pushing back

        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig = plt.figure(figsize=(8, 4))

        axs = fig.add_subplot(121)
        axs2 = fig.add_subplot(122)

        plotter.plot_ee_force_ts(axs)
        axs.xaxis.set_major_formatter(FuncFormatter(math_formatter))
        axs.yaxis.set_major_formatter(FuncFormatter(math_formatter))
        axs.set_xlabel("Time [s]")
        axs.set_ylabel("Force [N]")
        axs.xaxis.set_ticks(np.arange(0, 4, 1))
        axs.yaxis.set_ticks(np.arange(-60, 5, 20))
        axs.set_aspect("auto")
        axs.xaxis.label.set_fontsize(20)
        axs.yaxis.label.set_fontsize(20)
        axs.tick_params(axis="both", which="major", labelsize=20)
        axs.legend(loc="lower right", fontsize=22)

        plotter.plot_dist_ts(axs2)
        axs2.set_xlabel("Time [s]")
        axs2.set_ylabel("Distance [cm]")
        axs2.xaxis.set_major_formatter(FuncFormatter(math_formatter))
        axs2.yaxis.set_major_formatter(FuncFormatter(math_formatter))
        axs2.xaxis.set_ticks(np.arange(0, 4, 1))
        axs2.yaxis.set_ticks(np.arange(60, 76, 5))
        axs2.set_aspect("auto")
        axs2.xaxis.label.set_fontsize(20)
        axs2.yaxis.label.set_fontsize(20)
        axs2.tick_params(axis="both", which="major", labelsize=20)
        axs2.legend(loc="lower right", fontsize=19)

        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        plt.show()
        # plotter.save_fig("uc2_pushing_back_ts")


if __name__ == "__main__":
    uc1_run_dir = "freddy_uc1_log"
    uc2_run_dir = "freddy_uc2_align_log"

    uc1_plotter = UCPlotter(uc1_run_dir)
    uc1_plotter.plot_uc1_ts()

    # uc2_plotter = UCPlotter(uc2_run_dir)
    # uc2_plotter.plot_uc2_ts()
