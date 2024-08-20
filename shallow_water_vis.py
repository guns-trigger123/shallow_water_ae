from matplotlib import pyplot as plt
import numpy as np

data = np.load('data/R_15_Hp_6.npy', allow_pickle=True, mmap_mode='r')
array = data[50]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Titles for each channel
titles = ['u', 'v', 'h']

# Plot each channel in a separate subplot
for i in range(3):
    im = axes[i].imshow(array[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off')  # Hide the axes
    fig.colorbar(im, ax=axes[i], orientation='vertical')

# Display the plot
plt.show()