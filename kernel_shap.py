import numpy as np
import shap
from keras.models import load_model
from skimage.segmentation import slic
from processing_images import load_as_rgb_float
from plot_images import plot_heatmaps

# select the model and the dataset: (simple, complex, vgg16) (clean, kaggle1500)
model_name = 'complex'
dataset_name = 'kaggle1500'
class_names = ('health', 'tumor')

# load the pretrained model and the experiments (images).
model = load_model('./results/models/' + model_name + '_' + dataset_name + '.h5')

experiments_path = './datasets/brain_tumor_' + dataset_name + '/experiments/'
experiments_num = 10  # set the number of experiments per class

print("[INFO]: Experiments Loading")

images = load_as_rgb_float(experiments_path, [experiments_num * len(class_names), 256, 256, 3],
                           ('health/', 'tumor/'))


# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(mask, segmentation, img):
    back_grd = img.mean((0, 1))
    out = np.zeros((mask.shape[0], img.shape[0], img.shape[1], img.shape[2]))

    for i in range(mask.shape[0]):
        out[i, :, :, :] = img
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                out[i][segmentation == j, :] = back_grd

    return out

# fill the segment in plot
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


# use Kernel SHAP to explain test set predictions
print("[INFO]: explaining predictions.")

for idx, image in enumerate(images):


    # divide image in x segments. Then compute only x shap values and not one per pixel
    image_segments = slic(image, n_segments=50, compactness=30, sigma=3, start_label=1)

    # use Kernel SHAP to explain the network's predictions
    def predict_post_processing(mask):
        masked_image = mask_image(mask, image_segments, image)
        return model.predict(masked_image)

    explainer = shap.KernelExplainer(predict_post_processing, np.zeros((1, 50)))
    shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)

    prediction = model.predict(image.reshape((1,) + image.shape))[0]
    top_preds = np.argsort(-prediction)

    # heatmaps
    heatmap_top = fill_segmentation(shap_values[top_preds[0]][0], image_segments)
    heatmap_second = fill_segmentation(shap_values[top_preds[1]][0], image_segments)

    # plot our explanations
    max_value = max(max(abs(shap_values[top_preds[0]][0])), max(abs(shap_values[top_preds[1]][0])))

    original_class_index = int(idx / experiments_num)
    save_path = './results/kernel_exp/' + model_name + '_' + dataset_name \
                + '/' + class_names[original_class_index] + '_' \
                + str(idx % experiments_num) + '.png'

    plot_heatmaps(image, [heatmap_top, heatmap_second], class_names[original_class_index],
                  [class_names[top_preds[0]], class_names[top_preds[1]]],
                  [prediction[top_preds[0]], prediction[top_preds[1]]], max_value, save_path, 'SHAP'
                  )
