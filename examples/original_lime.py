import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import requests
import matplotlib.pylab as plt
import numpy as np
from lime import lime_image

# load model
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
model = VGG16()


# load images
images = np.zeros((7, 224, 224, 3))

for i, filename in enumerate(os.listdir('./original_images/')):

    img = image.load_img('./original_images/' + filename, target_size=(224, 224))
    images[i] = image.img_to_array(img)


explainer = lime_image.LimeImageExplainer()

def f(im):
    return model.predict(preprocess_input(im))


for image in images:
    explanation = explainer.explain_instance(image.astype('double'), f,
                                             top_labels=3, hide_color=None, num_samples=1000)


    # Plot. The visualization makes more sense if a symmetrical colorbar is used.
    preds = model.predict(preprocess_input(np.expand_dims(image.copy(), axis=0)))
    top_preds = np.argsort(-preds)[0]

    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(image/255)
    axes[0].axis('off')
    max_val = 0.0
    heatmaps = []

    for j in range(3):

        # Select the same class explained on the figures above.
        ind = explanation.top_labels[j]
        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmaps.append(np.vectorize(dict_heatmap.get)(explanation.segments))
        max_val = np.max([np.abs(heatmaps[j]).max(), max_val])

    for j in range(3):

        axes[j + 1].imshow(image/255, alpha=0.15)
        axes[j + 1].axis('off')
        im = axes[j + 1].imshow(heatmaps[j], cmap = 'RdBu', vmin = -max_val, vmax = max_val)
        axes[j + 1].set_title(feature_names[str(top_preds[j])][1])

    plt.colorbar(im, ax=axes.ravel().tolist(), label="LIME value", orientation="horizontal", aspect=60)
    plt.show()