import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import requests
from skimage.segmentation import slic
import numpy as np
import shap
from plot_images import plot_heatmaps

# load model data
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
model = VGG16()

# load images
images = np.zeros((7, 224, 224, 3))
names = []
for i, filename in enumerate(os.listdir('./original_images/')):

    img = image.load_img('./original_images/' + filename, target_size=(224, 224))
    images[i] = image.img_to_array(img)
    names.append(filename[0:len(filename)-4])

# function to mask images, taken from original_shap
def mask_image(mask, segmentation, im):
    back_grd = im.mean((0, 1))
    out = np.zeros((mask.shape[0], im.shape[0], im.shape[1], im.shape[2]))

    for j in range(mask.shape[0]):
        out[j, :, :, :] = im
        for k in range(mask.shape[1]):
            if mask[j, k] == 0:
                out[j][segmentation == k, :] = back_grd

    return out


# fill the segments in plot
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for j in range(len(values)):
        out[segmentation == j] = values[j]
    return out


for idx, image in enumerate(images):

    # divide image in 50 segments. Then compute only 50 shap values and not one per pixel
    image_segments = slic(image, n_segments=50, compactness=30, sigma=3, start_label=1)

    # use Kernel SHAP to explain the network's predictions
    def predict_post_processing(mask):
        masked_image = preprocess_input(mask_image(mask, image_segments, image))
        return model.predict(masked_image)

    explainer = shap.KernelExplainer(predict_post_processing, np.zeros((1, 50)))
    shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)

    # get the top predictions from the model
    preds = model.predict(preprocess_input(np.expand_dims(image.copy(), axis=0)))
    top_3_preds_idx = np.argsort(-preds)[0][0:3]

    # heatmaps
    heatmap_top = fill_segmentation(shap_values[top_3_preds_idx[0]][0], image_segments)
    heatmap_second = fill_segmentation(shap_values[top_3_preds_idx[1]][0], image_segments)
    heatmap_third = fill_segmentation(shap_values[top_3_preds_idx[2]][0], image_segments)

    # prepare data
    max_value = max(max(abs(shap_values[top_3_preds_idx[0]][0])),
                    max(abs(shap_values[top_3_preds_idx[1]][0])),
                    max(abs(shap_values[top_3_preds_idx[2]][0])))


    save_path = './my_kernel_shap/' + names[idx] + '.png'

    top_3_class_names = (feature_names[str(top_3_preds_idx[0])][1],
                         feature_names[str(top_3_preds_idx[1])][1],
                         feature_names[str(top_3_preds_idx[2])][1])
    top_3_preds = (preds[0][top_3_preds_idx[0]], preds[0][top_3_preds_idx[1]], preds[0][top_3_preds_idx[2]])

    plot_heatmaps(image/255, [heatmap_top, heatmap_second, heatmap_third], names[idx],
                  top_3_class_names, top_3_preds, max_value, save_path, 'SHAP'
                  )
