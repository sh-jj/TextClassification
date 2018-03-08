import numpy as np
from keras.models import Model, model_from_yaml
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from vgg_16 import VGG16
from aencoder import find_nbs
import matplotlib.pyplot as plt

base_model = VGG16(include_top=True, weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

img = image.load_img('test.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feature = model.predict(x)

encoder = model_from_yaml(open('encoder.yaml').read())
encoder.load_weights('encoder_weights.h5')

feature = encoder.predict(feature)
a = []
for i in feature:
    a = i
nbs = find_nbs(a, 5)
print(a.tolist())
plt.figure()
n = 1
for i in nbs:
    path = ('images/a/%d.jpg') % i
    img = image.load_img(path)
    plt.subplot(2,3,n)
    n+=1
    if n==2:
        plt.title('input image')
    else:
        plt.title('recommended image')
    plt.imshow(img)
plt.show()
print(nbs)
