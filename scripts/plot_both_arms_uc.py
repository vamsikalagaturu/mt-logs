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

    def plot_case_1_estimation_data(self):
        # run_id = "09_08_2024_18_48_50" # going back
        # 09_08_2024_19_48_02 # going forward
        # 09_08_2024_20_33_38 # sideways
        plotter = Plotter(self.run_dir)
        plotter.load_data(run_id)

        fig, axs = plotter.create_subplots(1, 1, (15, 15))

        # data_index = len(plotter.kr_df) // 2
        data_index = 0
        plotter.plot_ee_and_shoulder_lines(
            axs, data_index, colors=COLORS["initial"], linewidths=LWS["initial"]
        )
        center = plotter.get_base_center(data_index)
        plotter.plot_base_force_direction(axs, data_index, center)
        # plotter.plot_uc_data(axs, data_index)

        # plot the final condition
        data_index = len(plotter.kr_df) - 50
        plotter.plot_ee_and_shoulder_lines(
            axs,
            data_index,
            use_odometry=True,
            legend=False,
            colors=COLORS["final"],
            linewidths=LWS["final"],
        )
        # center = plotter.get_base_center(data_index)
        plotter.plot_base_force_direction(axs, data_index, center)
        # plotter.plot_uc_data(axs, data_index)

        # plotter.plot_base_wheel_coords(WHEEL_COORDINATES, axs)
        plotter.plot_base_odometry(axs, COLORS["final"])

        # aspect ratio of the plot
        axs.set_aspect("equal")
        axs.invert_xaxis()

        plt.show()


if __name__ == "__main__":
    run_dir = "freddy_uc1_log"
    both_arms_plotter = BothArmsPlotter(run_dir)
    # both_arms_plotter.plot_case_1_arms_data()
    both_arms_plotter.plot_case_1_estimation_data()
