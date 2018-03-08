#coding:utf-8
import sys
import keras
reload(sys)
sys.setdefaultencoding('utf8')

VECTOR_DIR = 'vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2

input_dim = 210

def get_input(path,dim0,dim):
	result = []
	print '---' 
	with open(path,"r") as f:
		for line in f:
		    item = line.split(' ')
		    result.append(map(float,item[dim0:dim]))
	#print(result)
	return result
	
print '(1) load texts...'
train_texts = get_input('dataset_vec/data_train.txt',0,210)
train_labels = get_input('dataset_vec/label_train.txt',0,1)
test_texts = get_input('dataset_vec/data_test.txt',0,210)
test_labels = get_input('dataset_vec/label_test.txt',0,1)
'''
print train_texts.shape
print train_labels.shape
print test_texts.shape
print test_labels.shape
'''

all_texts = train_texts + test_texts
all_labels = train_labels + test_labels

'''
feat_img = get_input('vec_img.txt')
feat_txt = get_input('vec_text.txt')
label = get_input('vec_text.txt')
'''

from keras.utils import to_categorical
import numpy as np

print '(3) split data set...'

data = np.asarray(all_texts)
labels = to_categorical(np.asarray(all_labels))

print labels

p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print 'train docs: '+str(len(x_train))
print 'val docs: '+str(len(x_val))
print 'test docs: '+str(len(x_test))

print '(5) training model...'
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.utils import plot_model

model = Sequential()
model.add(Dense(200, input_shape=(input_dim,), activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(200, input_shape=(200,), activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print model.metrics_names
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=6, batch_size=128)
model.save('mlp.h5')

print '(6) testing model...'
print model.evaluate(x_test, y_test)

        





