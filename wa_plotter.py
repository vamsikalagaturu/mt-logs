import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.ticker import FuncFormatter

# time = 54


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
        self.data_dir = "data"
        self.run_dir = run_dir

        self.wa_file = "wheel_align_log.csv"

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

        self.n = 4

        self.set_sns_props()

    def load_wa_data(self, run_id: str):
        self.run_id = run_id
        wa_file_path = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id, self.wa_file
        )
        self.wa_df = pd.read_csv(wa_file_path, index_col=False)
        self.save_dir = os.path.join(
            self.current_dir, self.data_dir, self.run_dir, run_id
        )

    def set_sns_props(self):
        sns.set_theme(style="whitegrid")
        sns.set_palette("deep")

    def plot_pivot(self, ax: plt.Axes, bilateral: bool = True):
        # get the data
        pivot_data = self.wa_df.filter(like="pivot")

        # x = np.linspace(0, time, len(kr_ee_s_dist))
        x = np.arange(len(pivot_data)) / 1000

        sns.lineplot(
            x=x,
            y=pivot_data["pivot_1"],
            ax=ax,
            label=r"$p_{1}$",
            color=gcolors["blue"],
            linewidth=2,
        )

    def plot_f_ref_null(self, ax: plt.Axes, bilateral: bool = True):
        # get the data
        f_drive_ref = self.wa_df.filter(like="f_drive_ref")
        f_null = self.wa_df.filter(like="f_null")
        f_null_scaled = self.wa_df.filter(like="f_null_scaled")
        # f_drive = self.wa_df.filter(like="f_drive")
        # tau_c = self.wa_df.filter(like="tau_c")

        # x = np.linspace(0, time, len(kr_ee_s_dist))
        x = np.arange(len(f_drive_ref)) / 1000

        sns.lineplot(
            x=x,
            y=f_drive_ref["f_drive_ref_2"],
            ax=ax,
            label=r"$f_{driveRef2}$",
            color=gcolors["pink"],
            linewidth=2,
        )
        sns.lineplot(
            x=x,
            y=f_null["f_null_1"],
            ax=ax,
            label=r"$f_{null2}$",
            color=gcolors["green"],
            linewidth=2,
        )

        ax2 = ax.twinx()

        sns.lineplot(
            x=x,
            y=f_null_scaled["f_null_scaled_2"],
            ax=ax2,
            label=r"$f_{nullScaled2}$",
            color=gcolors["red"],
            linewidth=2,
        )

    def save_fig(self, file_name: str, title: str = None, fontsize: int = 12):
        assert file_name is not None, "file_name cannot be None"

        # os.makedirs(self.save_dir, exist_ok=True)

        if title is not None:
            plt.suptitle(title, fontsize=fontsize)
        else:
            # remove title
            # plt.suptitle("")
            pf = self.wa_df.filter(like="pf_").iloc[0]
            plt.suptitle(
                f"pf = [{pf['pf_x']}, {pf['pf_y']}, {pf['pf_z']}]",
                fontsize=fontsize,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{file_name}.png"),
            format="png",
            transparent=False,
            pad_inches=0.0,
        )


class UCPlotter:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def plot_data(self):
        # run_id = "28_12_2024_15_58_57"
        run_id = "28_12_2024_15_59_09"
        # run_id = "28_12_2024_16_06_13"

        plotter = Plotter(self.run_dir)
        plotter.load_wa_data(run_id)

        fig = plt.figure(figsize=(8, 4))

        axs2 = fig.add_subplot(121)
        axs = fig.add_subplot(122)

        plotter.plot_pivot(axs)
        axs.set_xlabel("Time [s]")
        axs.set_ylabel("Pivot Angle [rad]")
        axs.set_aspect("auto")
        axs.tick_params(axis="both", which="major", labelsize=20)
        axs.legend(fontsize=20)
        # axs.xaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs.yaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs.xaxis.set_ticks(np.arange(0, 4, 1))
        # axs.yaxis.set_ticks(np.arange(-60, 30, 20))
        # axs.xaxis.label.set_fontsize(20)
        # axs.yaxis.label.set_fontsize(20)

        plotter.plot_f_ref_null(axs2)
        axs2.set_xlabel("Time [s]")
        axs2.set_ylabel("Force [N]")
        axs2.set_aspect("auto")
        axs2.tick_params(axis="both", which="major", labelsize=20)
        axs2.legend(fontsize=20)
        # axs2.xaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs2.yaxis.set_major_formatter(FuncFormatter(math_formatter))
        # axs2.xaxis.set_ticks(np.arange(0, 4, 1))
        # axs2.yaxis.set_ticks(np.arange(60, 85, 5))
        # axs2.xaxis.label.set_fontsize(20)
        # axs2.yaxis.label.set_fontsize(20)

        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        # plt.show()
        plotter.save_fig("wheel_align")


if __name__ == "__main__":
    wa_run_dir = "wheel_align_log"

    wa_plotter = UCPlotter(wa_run_dir)
    wa_plotter.plot_data()
