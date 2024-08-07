from plotter import Plotter, WHEEL_COORDINATES
import matplotlib.pyplot as plt
from utils import COLORS, LWS


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
        run_id = "07_08_2024_14_42_53"
        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig, axs = plotter.create_subplots(1, 1, (10, 10))

        # plot the initial condition
        data_index = 0
        plotter.plot_uc2_data(axs, data_index)
        plotter.plot_ee_and_shoulder_lines(
            axs, data_index, colors=COLORS["initial"], linewidths=LWS["initial"]
        )
        plotter.plot_wheel_pivot_directions(
            axs, data_index, WHEEL_COORDINATES, COLORS["initial"]["pivot"]
        )
        plotter.plot_base_wheel_coords(
            WHEEL_COORDINATES, axs, label="", color=COLORS["initial"]["base"]
        )

        # plot the final condition
        data_index = len(plotter.kr_df) - 1
        plotter.plot_ee_and_shoulder_lines(
            axs,
            data_index,
            use_odometry=True,
            legend=False,
            colors=COLORS["final"],
            linewidths=LWS["final"],
        )
        plotter.plot_base_odometry(axs, COLORS["final"])

        # axs.legend()

        axs.set_aspect("equal")
        axs.invert_xaxis()

        # plt.tight_layout()
        plt.show()
        # plotter.save_fig("uc2_pushing_back")

    def plot_pulling_forward_data(self):
        run_id = "07_08_2024_14_47_22"

        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig, axs = plotter.create_subplots(1, 1, (10, 10))

        # plot the initial condition
        data_index = 0
        plotter.plot_uc2_data(axs, data_index)
        plotter.plot_ee_and_shoulder_lines(
            axs, data_index, colors=COLORS["initial"], linewidths=LWS["initial"]
        )
        # plotter.plot_base_wheel_coords(WHEEL_COORDINATES, axs)

        # plot the final condition
        data_index = len(plotter.kr_df) - 1
        plotter.plot_ee_and_shoulder_lines(
            axs,
            data_index,
            use_odometry=True,
            legend=False,
            colors=COLORS["final"],
            linewidths=LWS["final"],
        )

        plotter.plot_base_odometry(axs)

        # axs.legend()

        axs.set_aspect("equal")
        axs.invert_xaxis()

        # plt.show()
        plotter.save_fig("uc2_pulling_forward")


if __name__ == "__main__":
    run_dir = "freddy_uc2_align_log"
    both_arms_plotter = BothArmsPlotter(run_dir)
    both_arms_plotter.plot_pushing_back_data()
    # both_arms_plotter.plot_pulling_forward_data()
