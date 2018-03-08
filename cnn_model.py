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


print '(1) load texts...'

#train_texts = open('train_contents.txt').read().split('\n')
#train_labels = open('train_labels.txt').read().split('\n')
#test_texts = open('test_contents.txt').read().split('\n')
#test_labels = open('test_labels.txt').read().split('\n')
all_texts = open('data_text.txt').read().split('\n')
all_labels = open('data_label.txt').read().split('\n')

print len(all_labels)
#print all_labels

print '(2) doc to var...'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


print 'shuffle...'

label_num = labels.shape[1]

training_data = np.hstack((data, labels))
np.random.shuffle(training_data)
data = training_data[:, :-label_num]
labels = training_data[:, -label_num:]


print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print '(3) split data set...'
# split the data into training set, validation set, and test set
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
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu', name='fc1'))
model.add(Dense(EMBEDDING_DIM, activation='relu', name='fc2'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print model.metrics_names
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=6, batch_size=128)
model.save('cnn_model_inferior.h5')

print '(6) testing model...'
print model.evaluate(x_test, y_test)

        




