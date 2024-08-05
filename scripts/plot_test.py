from utils import Plotter

if __name__ == "__main__":
    # create a plotter object
    plotter = Plotter("freddy_uc1_log_arm_test")

    # load the data
    plotter.load_data("05_08_2024_16_24_27")

    # plot the end effector orientation
    plotter.plot_ee_orientation(plotter.kr_df, "right", show=True)