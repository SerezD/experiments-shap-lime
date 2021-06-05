from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

def cmap():
    # make a color map for the plot
    colors = []
    for k in np.linspace(1, 0, 100):
        colors.append((245 / 255, 39 / 255, 87 / 255, k))

    for k in np.linspace(0, 1, 100):
        colors.append((24 / 255, 196 / 255, 93 / 255, k))
    return LinearSegmentedColormap.from_list("shap", colors)

def plot_heatmaps(original_image, heatmaps, original_class, class_names, predictions, max_value, save_path, flag):

    # display original image + all the heatmaps
    cols = len(heatmaps) + 1

    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(12, 6))

    # first axes is the original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original Image:\n' + str(original_class))

    for i in range(len(heatmaps)):

        axes[i + 1].set_title(class_names[i] + '\n {:.3f}'.format(predictions[i]))
        axes[i + 1].imshow(original_image, alpha=0.30)
        im = axes[i + 1].imshow(heatmaps[i], cmap=cmap(), vmin=-max_value, vmax=max_value)
        axes[i + 1].axis('off')

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label= flag + " values", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    plt.savefig(save_path)
    plt.close(fig)
