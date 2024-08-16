from plotter import Plotter, WHEEL_COORDINATES
import matplotlib.pyplot as plt
from utils import COLORS, LWS

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

from utils import make_legend_arrow


class BothArmsPlotter:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def plot_case_1_arms_data(self):
        run_id = "06_08_2024_15_37_21"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)
        plotter.load_kl_data(run_id)

        fig, axs = plotter.create_subplots(2, 2, (15, 15))

        plotter.plot_elbow_z_command_force(plotter.kr_df, axs[0][0], "right")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], "right")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], "right")

        plotter.plot_elbow_z_command_force(plotter.kl_df, axs[0][0], "left")
        plotter.plot_ee_orientation(plotter.kl_df, axs[0][1], "left")
        plotter.plot_elbow_and_ee_z(plotter.kl_df, axs[1][0], axs[1][1], "left")

        plt.show()

    def plot_case_1_estimation_data(self):
        # run_id = "09_08_2024_18_48_50"  # going back
        # run_id = "09_08_2024_16_22_25" # wheel alignment
        run_id = "09_08_2024_19_48_02" # going forward
        # run_id = "09_08_2024_20_33_38" # sideways
        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig, axs = plotter.create_subplots(1, 2, (30, 18))

        data_index = 0
        plotter.plot_ee_and_shoulder_lines(
            axs[0], data_index, colors=COLORS["initial"], linewidths=LWS["initial"]
        )
        center = plotter.get_base_center(WHEEL_COORDINATES)
        plotter.plot_base_wheel_coords(
            WHEEL_COORDINATES, axs[0], COLORS["initial"]["base"]
        )
        plotter.plot_uc_data(axs[0], data_index, colors=COLORS["initial"])
        plotter.plot_base_force_direction(
            axs[0], data_index, center, colors=COLORS["initial"]
        )
        # plot the pivot directions
        plotter.plot_wheel_pivot_directions(
            axs[0], data_index, WHEEL_COORDINATES, COLORS["initial"]["pivot"]
        )

        # plot the final condition
        data_index = len(plotter.kr_df) - 50
        plotter.plot_ee_and_shoulder_lines(
            axs[1],
            data_index,
            use_odometry=True,
            legend=True,
            colors=COLORS["initial"],
            linewidths=LWS["initial"],
        )
        # center = plotter.get_base_center(data_index)
        # plotter.plot_uc_data(axs, data_index)

        # plotter.plot_base_wheel_coords(WHEEL_COORDINATES, axs)
        plotter.plot_base_odometry(axs[1], COLORS["initial"], data_index)
        plotter.plot_uc_data(
            axs[1],
            data_index,
            colors=COLORS["initial"],
            use_odometry=True,
        )

        # set titles
        axs[0].set_title("Initial configuration")
        axs[1].set_title("Final configuration")

        # title font size
        axs[0].title.set_fontsize(23)
        axs[1].title.set_fontsize(23)

        # axis x, y labels
        axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        axs[1].set_xlabel("x [m]")
        axs[1].set_ylabel("y [m]")
        # axis x, y label font size
        axs[0].xaxis.label.set_fontsize(20)
        axs[0].yaxis.label.set_fontsize(20)
        axs[1].xaxis.label.set_fontsize(20)
        axs[1].yaxis.label.set_fontsize(20)

        # axis tick font size
        axs[0].tick_params(axis="both", which="major", labelsize=20)
        axs[1].tick_params(axis="both", which="major", labelsize=20)

        # set limits
        axs[0].set_xlim(-0.5, 0.5)
        axs[0].set_ylim(-0.25, 0.8)
        axs[1].set_xlim(-0.5, 0.5)
        axs[1].set_ylim(-0.25, 0.8)
        axs[0].set_aspect("equal")
        axs[1].set_aspect("equal")
        axs[0].invert_xaxis()
        axs[1].invert_xaxis()

        # plot the coordinate system
        # plot x axis
        plotter.plot_arrow(axs[0], -0.4, -0.2, 0.1, 0, "green")
        plotter.plot_arrow(axs[1], 0.3, 0.65, 0.1, 0, "green")
        # plot y axis
        plotter.plot_arrow(axs[0], -0.4, -0.2, 0, 0.1, "red")
        plotter.plot_arrow(axs[1], 0.3, 0.65, 0, 0.1, "red")
        # add text to the arrows
        axs[0].text(-0.425, -0.12, "x", fontsize=20)
        axs[0].text(-0.325, -0.23, "y", fontsize=20)

        axs[1].text(0.28, 0.73, "x", fontsize=20)
        axs[1].text(0.38, 0.62, "y", fontsize=20)

        # legend position
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(
            handles,
            labels,
            handler_map={
                mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
            },
            loc="upper left",
            fontsize=20,
        )
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(
            handles,
            labels,
            handler_map={
                mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
            },
            loc="lower left",
            fontsize=20,
            framealpha=0
        )

        # plt.show()
        plotter.save_fig(
            "uc1_forward",
            "Use Case 1: Alignment using active base control\n"
            "Base has to move forward to align and to maintain the target distance from the table",
            25,
        )

    def plot_base_force(self):
        run_id = "09_08_2024_20_33_38"  # going back
        # run_id = ""
        plotter2 = Plotter(self.run_dir)
        plotter2.load_uc_data(run_id)
        plotter2.load_mb_data(run_id)

        fig, axs = plotter2.create_subplots(1, 1, (15, 15))

        data_index = len(plotter2.mb_df) - 1
        plotter2.plot_base_force(axs, data_index, COLORS["initial"])

        axs.legend(loc="lower right", fontsize=20)

        # axis x, y label font size
        axs.xaxis.label.set_fontsize(20)
        axs.yaxis.label.set_fontsize(20)

        # axis tick font size
        axs.tick_params(axis="both", which="major", labelsize=20)

        plt.show()
        # plotter2.save_fig(
        #     "uc1_base_force_pivot_alignment_working",
        #     "Base force and pivot alignment control\n",
        #     25,
        # )

    def plot_base_force_and_pivot(self):
        # run_id = "09_08_2024_18_48_50"  # going back
        run_id = "09_08_2024_16_22_25" # wheel alignment
        # run_id = ""
        plotter2 = Plotter(self.run_dir)
        plotter2.load_uc_data(run_id)
        plotter2.load_mb_data(run_id)

        fig, axs = plotter2.create_subplots(1, 2, (30, 15))

        data_index = len(plotter2.mb_df) - 1
        plotter2.plot_base_force(axs[0], data_index, COLORS["initial"])
        plotter2.plot_pivot_direction(axs[1], 1)

        axs[0].legend(loc="lower right", fontsize=20)
        axs[1].legend(loc="lower right", fontsize=20)

        # axis x, y label font size
        axs[0].xaxis.label.set_fontsize(20)
        axs[0].yaxis.label.set_fontsize(20)
        axs[1].xaxis.label.set_fontsize(20)
        axs[1].yaxis.label.set_fontsize(20)

        # axis tick font size
        axs[0].tick_params(axis="both", which="major", labelsize=20)
        axs[1].tick_params(axis="both", which="major", labelsize=20)

        # plt.show()
        plotter2.save_fig(
            "uc1_base_force_pivot_alignment_working",
            "Base force and pivot alignment control\n",
            25,
        )


if __name__ == "__main__":
    run_dir = "freddy_uc1_log"
    both_arms_plotter = BothArmsPlotter(run_dir)
    # both_arms_plotter.plot_case_1_arms_data()
    # both_arms_plotter.plot_case_1_estimation_data()
    both_arms_plotter.plot_base_force()
