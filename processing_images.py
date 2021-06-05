import os
import numpy as np
from PIL import Image

def load_as_rgb_float(path, dest_array_shape, sub_dirs=()):

    result = np.zeros(dest_array_shape, dtype=np.float32)

    if len(sub_dirs) == 0:

        for image_index, filename in enumerate(os.listdir(path)):
            im = Image.open(path + filename)
            result[image_index] = np.asarray(im.convert('RGB'), dtype=np.float32) / 255

    else:
        image_index = 0
        for sub_dir in sub_dirs:

            extended_path = path + sub_dir

            for filename in os.listdir(extended_path):
                im = Image.open(extended_path + filename)

                result[image_index] = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
                image_index += 1

    return result
