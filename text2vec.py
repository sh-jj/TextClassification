#coding:utf-8
import sys
import keras
reload(sys)
sys.setdefaultencoding('utf8')


MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential


from keras.models import load_model

print '(1) load texts...'
'''
train_texts = open('train_contents.txt').read().split('\n')
train_labels = open('train_labels.txt').read().split('\n')
test_texts = open('test_contents.txt').read().split('\n')
test_labels = open('test_labels.txt').read().split('\n')
all_texts = train_texts + test_texts
all_labels = train_labels + test_labels
'''
all_texts = open('data_text.txt').read().split('\n')

all_labels = open('data_label.txt').read().split('\n')

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





from keras.models import Model

base_model=load_model('cnn_model.h5');
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

print labels.shape
print 'evalute...'
print base_model.evaluate(data,labels)


print 'model predict...'

output = model.predict(data)

print 'vec_shape' + str(output.shape)

with open('vec_text.txt','w') as f:
	for item in output:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')

print 'predict finish'


