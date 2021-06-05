# EXPERIMENTS WITH SHAP AND LIME

This code contains some experiments made with two state-of-the-art methods of explainable machine learning.

The final goal is to confront two different CNNs by explaining their results on a dataset made of brain tumor MRI. 

In particular, I used two different datasets for my experiments.

The first is composed of 3000 images, from the two classes "health" and "tumor", and data augmentation was computed.
I downloaded it from kaggle at this link: https://www.kaggle.com/abhranta/brain-tumor-detection-mri/metadata

The second one is a cleaned version of it, in which I manually removed artifacts obtaining 600 images (300 per class).

Both of the datasets are available in the homonymous directory.

### prepare dataset
This script contains the preprocessing operations that I computed on data. 

If you want to run it, remember to manually modify the path-variables indicating the images to preprocess.

The operations that will be computed are:
- duplicates removal.
- squaring of images, by extending borders with black colour.
- resizing of images to 256 x 256
- renaming of images.

### processing images and plot images
Those two scripts simply contain the functions I used in other classes.

### cnn
This script will train one of the two CNNs I used.
You can choose the dataset ('clean' or 'kaggle1500') and the net ('complex' or 'vgg16').

'complex' CNN is the one I created from scratch, while 'vgg16' is a fine-tuned version of the famous CNN.

Both models and training plots are already available in the "results" directory.

### lime_explanation and kernel_shap
These two classes produce the explanations for the networks and save them in the "results" directory. Again, you can set the correct parameters between 'clean' or 'kaggle1500' for datasets and 'complex' or 'vgg16' for CNNs.

Check also the path from which images will be loaded and the final path that will contain the results.

### examples
This directory contains a comparison between my lime and shap and the original's. This was made to be sure that my explanations work correctly, since I slightly modified some parts (for example, the plot function).

Examples were run taking the pretrained vgg16 network and classifying some random images. Then the explanations (original and mine) were produced and compared to check differences.
