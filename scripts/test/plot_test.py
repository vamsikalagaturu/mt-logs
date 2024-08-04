# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
import os

current_dir = os.path.dirname(__file__)

dir = 'data'
subdir = 'freddy_uc1_test'
save_dir = f'plots/{subdir}'
file_name1 = 'kinova_left_voltage_current_log.csv'
file_name2 = 'kinova_right_voltage_current_log.csv'
file_name3 = 'mobile_base_voltage_current_log.csv'

run_ids = os.listdir(os.path.join(current_dir, dir, subdir))

# run_id format: dd_mm_yyyy_hh_mm_ss
# sort run_ids
run_ids = sorted(run_ids)
run_id = run_ids[-1]
file_path1 = os.path.join(current_dir, dir, subdir, run_id, file_name1)
file_path2 = os.path.join(current_dir, dir, subdir, run_id, file_name2)
file_path3 = os.path.join(current_dir, dir, subdir, run_id, file_name3)

# read csv
# df1 = pd.read_csv('/home/batsy/rc/logs/data/freddy_uc1_test/22_07_2024_10_22_41/kinova_left_voltage_current_log.csv', index_col=0)
# df1 = pd.read_csv(file_path1, index_col=0)

print(file_path1)
print('/home/batsy/rc/logs/data/freddy_uc1_test/22_07_2024_10_22_41/kinova_left_voltage_current_log.csv')