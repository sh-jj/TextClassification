import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, model_from_yaml
from keras.layers import Dense, Input
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


def train_aencoder(feature_input):
    np.random.seed(1337)
    ae_input = Input(shape=(4096,))

    encoded = Dense(1024, activation='relu')(ae_input)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    encoder_output = Dense(10)(encoded)

    decoded = Dense(32, activation='relu')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(4096, activation='relu')(decoded)

    autoencoder = Model(input=ae_input, output=decoded)

    encoder = Model(input=ae_input, output=encoder_output)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    autoencoder.fit(feature_input, feature_input, epochs=100, batch_size=200, shuffle=True)

    ae = autoencoder.to_yaml()
    with open('autoencoder.yaml', 'w') as f1:
        f1.write(ae)
    autoencoder.save_weights('autoencoder_weights.h5')

    en = encoder.to_yaml()
    with open('encoder.yaml', 'w') as f2:
        f2.write(en)
    encoder.save_weights('encoder_weights.h5')

    return encoder


def write_embed(feature_input):
    encoder = model_from_yaml(open('encoder.yaml').read())
    encoder.load_weights('encoder_weights.h5')
    img_embedding = encoder.predict(feature_input)
    with open('img_embedding_pre.txt', 'w') as f:
        for i in img_embedding:
            for j in i:
                f.write(str(j) + ' ')
            f.write('\n')

'''
def find_nbs(input_embedding, neighbor_num):
    img_embedding = np.loadtxt('img_embedding1.txt').tolist()
    nbs = sorted(img_embedding, key=lambda nb:cosine(nb, input_embedding), reverse=True)
    nbs = nbs[:neighbor_num]
    nbs_index = []
    for nb in nbs:
        nbs_index.append(img_embedding.index(nb))
    return nbs_index
'''

def cosine(list1, list2):
    return cosine_similarity([list1],[list2])[0][0]


'''
if __name__ == '__main__':
    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype('float32') / 255. - 0.5

    x_test = x_test.reshape((x_test.shape[0], -1))

    print('load data...')
    # encoder = train_autoencoder(x_test)
    encoder = model_from_yaml(open('encoder.yaml').read())
    encoder.load_weights('encoder_weights.h5')

    ind = find_nbs([0.87314, -0.550542, 0.491887, 0.267009, 2.54022, -2.52929, -1.85952, 3.32853, -0.936145, -0.635357], 3)

    print(ind)
'''
