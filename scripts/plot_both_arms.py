from plotter import Plotter
import matplotlib.pyplot as plt


class BothArmsPlotter:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def plot_case_1_arms_data(self):
        run_id = "06_08_2024_13_16_41"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)
        plotter.load_kl_data(run_id)

        fig, axs = plotter.create_subplots(2, 2, (15, 15))

        # plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], "xy", "right")
        plotter.plot_elbow_z_command_force(plotter.kr_df, axs[0][0], "right")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], "right")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], "right")

        # plotter.plot_arm_trajectory(plotter.kl_df, axs[0][0], "xy", "left")
        plotter.plot_elbow_z_command_force(plotter.kl_df, axs[0][0], "left")
        plotter.plot_ee_orientation(plotter.kl_df, axs[0][1], "left")
        plotter.plot_elbow_and_ee_z(plotter.kl_df, axs[1][0], axs[1][1], "left")

        # plt.show()
        plotter.save_fig("both_arms_control",
        "Both Arms Control\n \
        EE Feed-forward Force Control, Position Control in Linear Y and Z, and Orientation Control & \n \
        Elbow Distance Control")
                         

    def plot_case_1_estimation_data(self):
        run_id = "06_08_2024_13_16_41"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)
        plotter.load_kl_data(run_id)

        fig, axs = plotter.create_subplots(1, 1, (15, 15))

        data_index = len(plotter.kr_df) // 2
        plotter.plot_ee_and_shoulder_lines(axs, data_index)
        plotter.plot_base_wheel_coords(axs)

        # aspect ratio of the plot
        axs.set_aspect("equal")
        axs.invert_xaxis()

        plt.show()



if __name__ == "__main__":
    run_dir = "freddy_uc1_log_both_arms_test"
    both_arms_plotter = BothArmsPlotter(run_dir)
    both_arms_plotter.plot_case_1_arms_data()
    # both_arms_plotter.plot_case_1_estimation_data()
