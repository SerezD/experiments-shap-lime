import numpy as np
from keras.models import load_model
from lime import lime_image
from processing_images import load_as_rgb_float
from plot_images import plot_heatmaps


# select the model and the dataset: (complex, vgg16) (clean, kaggle1500)
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

# explanations:
print("[INFO]: analyzing the explanations:\n")

# define the explainer
explainer = lime_image.LimeImageExplainer()

for im_idx, image in enumerate(images):


    explanation = explainer.explain_instance(image.astype('double'), model.predict,
                                             top_labels=2, hide_color=None, num_samples=1000)

    prediction = model.predict(image.reshape((1,) + image.shape))[0]
    top_preds = np.argsort(-prediction)

    max_value = 0.0
    heatmaps = []

    for j in range(len(class_names)):

        ind = explanation.top_labels[j]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmaps.append(np.vectorize(dict_heatmap.get)(explanation.segments))
        max_value = np.max([np.abs(heatmaps[j]).max(), max_value])

    if max_value < 0.0001:
        max_value = 0.0001

    # plot images
    original_class_index = int(im_idx / experiments_num)
    save_path = './results/lime_exp/' + model_name + '_' + dataset_name \
                + '/' + class_names[original_class_index] + '_' \
                + str(im_idx % experiments_num) + '.png'

    plot_heatmaps(image, heatmaps, class_names[original_class_index],
                  [class_names[top_preds[0]], class_names[top_preds[1]]],
                  [prediction[top_preds[0]], prediction[top_preds[1]]], max_value, save_path, 'LIME'
                  )
