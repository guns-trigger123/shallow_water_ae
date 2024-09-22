from matplotlib import pyplot as plt
import numpy as np

R, Hp, timestep = 20, 12, 100
data = np.load(f'../data/train/raw/R_{R}_Hp_{Hp}.npy', allow_pickle=True, mmap_mode='r')
# data_conservative = np.load('../data/R_10_Hp_5_conservative.npy', allow_pickle=True, mmap_mode='r')
# data = np.load("../data/R_10_Hp_5.npy", allow_pickle=True, mmap_mode='r')
array = data[timestep]
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Titles for each channel
titles = ['u', 'v', 'h']

# Plot each channel in a separate subplot
for i in range(3):
    im = axes[i].imshow(array[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off')  # Hide the axes
    fig.colorbar(im, ax=axes[i], orientation='vertical')

# Display the plot
# fig.suptitle(f"R = {R} Hp = {Hp} timestep = {timestep}")
fig.savefig(f"./R = {R} Hp = {Hp} timestep = {timestep}.pdf", format='pdf')
plt.show()