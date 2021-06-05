from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.applications import VGG16
from processing_images import load_as_rgb_float
from matplotlib import pyplot as plt

# Conv models used ('complex', 'vgg16')
def set_conv_model(model_type):

    # add model layers
    if model_type == 'complex':

        model = Sequential()

        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))

    elif model_type == 'vgg16':

        vgg_16 = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

        # freeze every layer in our model so that they do not train
        for layer in vgg_16.layers:
            layer.trainable = False

        model = Sequential()
        # Add the vgg convolutional base model
        model.add(vgg_16)

        # Add new layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

    else:
        raise Exception("The model Name is not correct")

    # model summary
    print(model.summary())

    return model


# Load data
print("[INFO]: Dataset Loading")

# choose dataset between "clean" and "kaggle1500"
dataset_flag = 'kaggle1500'

# choose model between 'complex' or 'vgg16'
m_flag = 'complex'

test_path = './datasets/brain_tumor_' + dataset_flag + '/test/'
train_path = './datasets/brain_tumor_' + dataset_flag + '/train/'

# 0 = health, 1 = tumor
class_names = ('health', 'tumor')

# dataset clean -->  [550, 256, 256] + [100, 256, 256]
# dataset kaggle1500 --> [2600, 256, 256] + [300, 256, 256]
if dataset_flag == 'clean':
    train_num, test_num = 550, 100
else:
    train_num, test_num = 2600, 300

# load_data
X_train = load_as_rgb_float(train_path, [train_num, 256, 256, 3], ('health/', 'tumor/'))
X_test = load_as_rgb_float(test_path, [test_num, 256, 256, 3], ('health/', 'tumor/'))

y_train = to_categorical([0] * int(train_num/2) + [1] * int(train_num/2))
y_test = to_categorical([0] * int(test_num/2) + [1] * int(test_num/2))

# create the model
m = set_conv_model(m_flag)

# compile model
# Adam optimizer for learning rate
# categorical_cross_entropy as loss function
# accuracy to measure model performance
m.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
earl = EarlyStopping(monitor="loss", patience=2, mode="min")

history = m.fit(X_train, y_train, batch_size=50, validation_data=(X_test, y_test),
                epochs=100, callbacks=earl)

# plot results
fig, (up, down) = plt.subplots(2)
up.plot(history.history['accuracy'], label='accuracy')
up.plot(history.history['val_accuracy'], label = 'val_accuracy')
up.set_xlabel('Epoch')
up.set_ylabel('Accuracy')
up.set_ylim([0, 1])
up.legend(loc='lower right')

down.plot(history.history['loss'], label = 'loss')
down.plot(history.history['val_loss'], label = 'val_loss')
down.set_xlabel('Epoch')
down.set_ylabel('Loss')
down.set_ylim([0, 3])
down.legend(loc='lower right')

plt.suptitle('Training Progress for Model ' + m_flag + ' on Dataset ' + dataset_flag)
plt.savefig('./results/plots/' + m_flag + '_' + dataset_flag + '.png')
plt.show()

m.save('./results/models/' + m_flag + '_' + dataset_flag + '.h5')
