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
#from aencoder import find_nbs
import matplotlib.pyplot as plt
import os


numer = [368,308,290,123,303,351,320,214]
#numer = [10,10,10,10,10,10,10,10]

def get_img_input():
    img_input = []

    for i in range(8):
        for j in range(numer[i]):
            path = ("patch/" + str(i) + "/data/" + str(j) +"/")
            if os.path.exists(path + '0.jpg'):
                img_path = path + '0.jpg'
            else:
                if os.path.exists(path + '0.jpeg'):
                    img_path = path + '0.jpeg'
                else:
                    if os.path.exists(path + '0.png'):
                        img_path = path + '0.png'
                    else:
                        img_path = 'default.jpg'
            
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            img_input.append(x)
            print 'loading...' + 'class ' + str(i) + ' ' + str(j+1) + '/' +str(numer[i])
            
            
    '''
    for x in range(img_num):
        img_path = ('test/%d.jpg') % x
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        img_input.append(x)
    '''
    
    print 'loading finished'
    return img_input



base_model = VGG16(include_top=True, weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

'''
    base_model = VGG16(include_top=True, weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

    n = 0
    img_num = 400
    img_input = get_img_input(img_num)
    img_feature = np.ndarray([1,4096])
    for i in img_input:
        n += 1
        print('processing {}/{} '.format(n, img_num))
        feature = model.predict(i)
        #feature = feature.astype('float32') / 255. - 0.5
        img_feature = np.concatenate((img_feature, feature), axis=0)

    img_feature = img_feature[1:]
    print(img_feature.shape)
    autoencoder = aencoder.train_aencoder(img_feature)
    print('ae train done writing...')
    aencoder.write_embed(img_feature)
'''

n = 0
img_input = get_img_input()

img_num = len(img_input)
print 'img_num: ' + str(img_num)

img_feature = np.ndarray([1,4096])
for i in img_input:
	n += 1
	print('processing {}/{} '.format(n, img_num))
	feature = model.predict(i)
	#feature = feature.astype('float32') / 255. - 0.5
	img_feature = np.concatenate((img_feature, feature), axis=0)

img_feature = img_feature[1:]
print(img_feature.shape)
encoder = model_from_yaml(open('encoder.yaml').read())
encoder.load_weights('encoder_weights.h5')

feature_set = encoder.predict(img_feature)
	
with open('vec_all.txt', 'w') as f:
	for i in feature_set:
		for j in i:
			f.write(str(j) + ' ')
		f.write('\n')

print 'finish'
