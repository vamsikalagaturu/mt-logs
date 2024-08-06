from plotter import Plotter
import matplotlib.pyplot as plt

class SingleArmPlotter:
    def __init__(self, run_dir: str) -> None:

        self.run_dir = run_dir

    def plot_case_1(self):
        run_id = "05_08_2024_13_06_48"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_1_1(self):
        run_id = "05_08_2024_13_39_21"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_1_2(self):
        run_id = "05_08_2024_13_41_23"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_2(self):
        run_id = "05_08_2024_13_54_53"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_3(self):
        run_id = "05_08_2024_14_40_32"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_4(self):
        run_id = "05_08_2024_15_17_21"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_5(self):
        run_id = "05_08_2024_15_28_58"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_6(self):
        run_id = "05_08_2024_15_32_30"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_7(self):
        run_id = "05_08_2024_15_37_07"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_8(self):
        run_id = "05_08_2024_16_18_39"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

    def plot_case_9(self):
        run_id = "05_08_2024_16_24_27"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        # plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy")
        plotter.plot_elbow_z_command_force(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1])
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1])
        plt.show()

if __name__ == "__main__":
    # create a plotter object
    plotter = SingleArmPlotter("freddy_uc1_log_arm_test")
    # plotter.plot_case_1()
    # plotter.plot_case_1_1()
    # plotter.plot_case_1_2()
    # plotter.plot_case_2()
    # plotter.plot_case_3()
    # plotter.plot_case_4()
    # plotter.plot_case_5()
    # plotter.plot_case_6()
    # plotter.plot_case_7()
    # plotter.plot_case_8()
    # plotter.plot_case_9()
