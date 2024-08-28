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
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], prefix="a")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], prefix="a")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], prefix="c,d")
        plt.show()

    def plot_case_2(self):
        run_id = "05_08_2024_13_54_53"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], prefix="a")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], prefix="b")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], prefix="c,d")
        # plt.show()
        plotter.save_fig("ms1_sa", 
        "Motion Specification for Single Arm\n \
        Feed-forward Force Control and Position Control")

    def plot_case_3(self):
        run_id = "05_08_2024_14_40_32"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy", prefix="a")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], prefix="b")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], prefix="c,d")
        # plt.show()
        plotter.save_fig("ms2_sa",
        "Motion Specification for Single Arm\n \
        EE Feed-forward Force Control and Position Control & Elbow Distance Control")

    def plot_case_4(self):
        run_id = "05_08_2024_15_17_21"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the elbow and end effector z position
        fig, axs = plotter.create_subplots(2, 2, (15, 15))
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy", prefix="a")
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], prefix="b")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], prefix="c,d")
        # plt.show()
        plotter.save_fig("ms3_sa",
        "Motion Specification for Single Arm\n \
        EE Feed-forward Force Control and Position Control in Linear Y and Z & Elbow Distance Control")

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
        plotter.plot_arm_trajectory(plotter.kr_df, axs[0][0], coord="xy", prefix="a")
        # plotter.plot_elbow_z_command_force(plotter.kr_df, axs[0][0])
        plotter.plot_ee_orientation(plotter.kr_df, axs[0][1], prefix="b")
        plotter.plot_elbow_and_ee_z(plotter.kr_df, axs[1][0], axs[1][1], prefix="c,d")
        # plt.show()
        plotter.save_fig("ms4_sa",
        "Motion Specification for Single Arm\n \
        EE Feed-forward Force Control, Position Control in Linear Y and Z, and Orientation Control & \n \
        Elbow Distance Control")

    def plot_no_gravity_comp(self):
        run_id = "23_08_2024_19_02_55"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the end effector z position
        fig, axs = plotter.create_subplots(1, 1, (15, 15))
        plotter.plot_ee_z(plotter.kr_df, axs)

        axs.legend(loc="lower left", fontsize=40)
        axs.xaxis.label.set_fontsize(40)
        axs.yaxis.label.set_fontsize(40)
        axs.tick_params(axis="both", which="major", labelsize=40)

        # plt.show()
        plotter.save_fig("no_gravity_comp", None)


    def plot_gravity_comp(self):
        run_id = "23_08_2024_19_06_06"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the end effector z position
        fig, axs = plotter.create_subplots(1, 1, (15, 15))
        plotter.plot_ee_z(plotter.kr_df, axs)

        axs.legend(loc="lower right", fontsize=40)
        axs.xaxis.label.set_fontsize(40)
        axs.yaxis.label.set_fontsize(40)
        axs.tick_params(axis="both", which="major", labelsize=40)

        # plt.show()
        plotter.save_fig("gravity_comp", None)

    def plot_elbow_z(self):
        run_id = "23_08_2024_18_54_44"
        plotter = Plotter(self.run_dir)
        plotter.load_kr_data(run_id)

        # plot the end effector z position
        fig, axs = plotter.create_subplots(1, 1, (15, 15))
        plotter.plot_elbow_z(plotter.kr_df, axs)

        axs.legend(loc="lower right", fontsize=40)
        axs.xaxis.label.set_fontsize(40)
        axs.yaxis.label.set_fontsize(40)
        axs.tick_params(axis="both", which="major", labelsize=40)

        # plt.show()
        plotter.save_fig("elbow_spring", None)

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
    # plotter.plot_no_gravity_comp()
    # plotter.plot_gravity_comp()
    plotter.plot_elbow_z()
