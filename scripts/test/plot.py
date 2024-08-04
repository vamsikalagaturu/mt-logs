import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(__file__)

dir = 'data'
subdir = 'kr_uc1_world'
save_dir = f'plots/{subdir}'
file_name = 'mobile_base_log.csv'

run_ids = os.listdir(os.path.join(current_dir, dir, subdir))

# run_id format: dd_mm_yyyy_hh_mm_ss
# sort run_ids
run_ids = sorted(run_ids)

# save all runs
run_id = run_ids[-1]
file_path = os.path.join(current_dir, dir, subdir, run_id, file_name)

# read csv
df = pd.read_csv(file_path, index_col=False)

# use x axis as length of the data
x = np.arange(len(df))

# plot xd_platform_x, xd_platform_y, xd_platform_qz
# subplots 

fig, axs = plt.subplots(2, 3, figsize=(10, 10))

# xd_platform_x
axs[0, 0].plot(x, df['xd_platform_x'], label='xd_platform_x')
axs[0, 0].set_title('xd_platform_x')
axs[0, 0].legend()

# xd_platform_y
axs[0, 1].plot(x, df['xd_platform_y'], label='xd_platform_y')
axs[0, 1].set_title('xd_platform_y')
axs[0, 1].legend()

# xd_platform_qz
axs[0, 2].plot(x, df['xd_platform_qz'], label='xd_platform_qz')
axs[0, 2].set_title('xd_platform_qz')
axs[0, 2].legend()

# x_platform_x vs x_platform_y
axs[1, 0].plot(df['x_platform_x'], df['x_platform_y'], label='x_platform_x vs x_platform_y')
axs[1, 0].set_title('x_platform_x vs x_platform_y')
axs[1, 0].legend()

# x_platform_qz
axs[1, 1].plot(x, df['x_platform_qz'], label='x_platform_qz')
axs[1, 1].set_title('x_platform_qz')
axs[1, 1].legend()

# save plot
# save_path = os.path.join(current_dir, save_dir, run_id)
# os.makedirs(save_path, exist_ok=True)
# plt.savefig(os.path.join(save_path, 'plot.png'))

# # plot title
plt.suptitle(f'Run ID: {run_id}')

plt.show()

