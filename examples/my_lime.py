import os
import requests
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from lime import lime_image
from plot_images import plot_heatmaps


# load model data
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
model = VGG16()


# load an image
images = np.zeros((7, 224, 224, 3))
names = []

for i, filename in enumerate(os.listdir('./original_images/')):

    img = image.load_img('./original_images/' + filename, target_size=(224, 224))
    images[i] = image.img_to_array(img)
    names.append(filename[0:len(filename) - 4])

# initialize the explainer
explainer = lime_image.LimeImageExplainer()

# there is the need to preprocess image before to make the prediction
def f(im):
    return model.predict(preprocess_input(im))


for idx, image in enumerate(images):

    # make the explanation returning top 3 labels.
    explanation = explainer.explain_instance(image.astype('double'), f, top_labels=3, hide_color=None,
                                             num_samples=1000)

    # Make the predictions and extract top predictions
    preds = model.predict(preprocess_input(np.expand_dims(image.copy(), axis=0)))
    top_preds = np.argsort(-preds)[0]

    # prepare data
    max_val = 0.0
    heatmaps = []

    for j in range(3):

        ind = explanation.top_labels[j]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmaps.append(np.vectorize(dict_heatmap.get)(explanation.segments))
        max_val = np.max([np.abs(heatmaps[j]).max(), max_val])

    class_names = (feature_names[str(top_preds[0])][1],
                   feature_names[str(top_preds[1])][1],
                   feature_names[str(top_preds[2])][1])
    top_3_preds = (preds[0][top_preds[0]], preds[0][top_preds[1]], preds[0][top_preds[2]])

    save_path = './results/my_lime/' + names[idx] + '.png'

    # plot function
    plot_heatmaps(image/255, heatmaps, names[idx], class_names,
                  top_3_preds, max_val, save_path, "LIME"
                  )
