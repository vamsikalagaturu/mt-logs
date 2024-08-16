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

    def plot_pushing_back_data(self):
        # run_id = "07_08_2024_14_42_53"  # pushing back
        # run_id = "07_08_2024_14_47_22" # pulling forward
        run_id = "07_08_2024_14_54_32"  # sideways 07_08_2024_14_54_32

        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig, axs = plotter.create_subplots(1, 2, (30, 15))

        # plot the initial condition
        data_index = 0
        plotter.plot_ee_and_shoulder_lines(
            axs[0], data_index, colors=COLORS["initial"], linewidths=LWS["initial"]
        )
        plotter.plot_uc2_data(axs[0], data_index, colors=COLORS["initial"])
        plotter.plot_wheel_pivot_directions(
            axs[0], data_index, WHEEL_COORDINATES, COLORS["initial"]["pivot"]
        )
        plotter.plot_base_wheel_coords(
            WHEEL_COORDINATES, axs[0], label="", color=COLORS["initial"]["base"]
        )

        # plot the final condition
        data_index = len(plotter.kr_df) - 1
        plotter.plot_ee_and_shoulder_lines(
            axs[1],
            data_index,
            use_odometry=True,
            legend=True,
            colors=COLORS["initial"],
            linewidths=LWS["initial"],
        )
        plotter.plot_base_odometry(axs[1], COLORS["initial"], data_index)
        # plotter.plot_uc2_data(
        #     axs[1],
        #     data_index,
        #     colors=COLORS["initial"],
        #     use_odometry=True,
        # )

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

        axs[0].set_xlim(-0.8, 0.3)
        axs[0].set_ylim(-0.6, 0.6)
        axs[1].set_xlim(-0.8, 0.3)
        axs[1].set_ylim(-0.6, 0.6)

        axs[0].set_aspect("equal")
        axs[1].set_aspect("equal")
        axs[0].invert_xaxis()
        axs[1].invert_xaxis()

        # axis tick font size
        axs[0].tick_params(axis="both", which="major", labelsize=20)
        axs[1].tick_params(axis="both", which="major", labelsize=20)

        # legend position
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(
            handles,
            labels,
            handler_map={
                mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
            },
            loc="lower left",
            fontsize=20,
            framealpha=0.5,
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
            framealpha=0.5,
        )

        # plt.show()
        # plotter.save_fig(
        #     "uc2_pushing_back",
        #     "Use Case 2: Alignment by pushing the table with both arms",
        #     25,
        # )
        # plotter.save_fig("uc2_pulling_forward",
        # "Use Case 2: Alignment by pulling the table with both arms", 25)
        plotter.save_fig(
            "uc2_sideways",
            "Use Case 2: Alignment by pushing/pulling the table with both arms to result in sideways motion",
            25,
        )

    def plot_base_force_and_pivot(self):
        # run_id = "07_08_2024_14_24_08"
        # run_id = "07_08_2024_14_30_50"
        # run_id = "07_08_2024_14_32_27"
        run_id = "07_08_2024_14_34_21"
        # run_id = "07_08_2024_14_42_53"
        # run_id = "07_08_2024_14_47_22"

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
            "uc2_manual_force",
            "Use Case 2: Testing manually by exerting pushing/pulling force on the end-effectors",
            25,
        )


if __name__ == "__main__":
    run_dir = "freddy_uc2_align_log"
    both_arms_plotter = BothArmsPlotter(run_dir)
    both_arms_plotter.plot_pushing_back_data()
    # both_arms_plotter.plot_base_force_and_pivot()
