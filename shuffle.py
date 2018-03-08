

VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2

def get_input(path,dim):
	result = []
	print '---' 
	with open(path,"r") as f:
		for line in f:
		    item = line.split(' ')
		    result.append(map(float,item[:dim]))
	#print(result)
	return result


feat_img = get_input('vec_img.txt',10)
feat_txt = get_input('vec_text.txt',200)
label = get_input('data_label.txt',1)


import numpy as np


feat_img = np.asarray(feat_img)
feat_txt = np.asarray(feat_txt)
feat_label = np.asarray(label)

print feat_img.shape
print feat_txt.shape
print feat_label.shape

from sklearn import preprocessing
feat_img = preprocessing.scale(feat_img)
feat_txt = preprocessing.scale(feat_txt)

alls = np.hstack((feat_img, feat_txt))
alls = np.hstack((alls, feat_label))

#print alls
print alls.shape

np.random.shuffle(alls)
#print alls

data = alls[:, :-1]
labels = alls[:, -1:]

with open('dataset_vec/data_shuffle.txt','w') as f:
	for item in data:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')

with open('dataset_vec/label_shuffle.txt','w') as f:
	for item in labels:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')
		
p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p2]
y_train = labels[:p2]
x_test = data[p2:]
y_test = labels[p2:]

print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

with open('dataset_vec/data_train.txt','w') as f:
	for item in x_train:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')

with open('dataset_vec/label_train.txt','w') as f:
	for item in y_train:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')

with open('dataset_vec/data_test.txt','w') as f:
	for item in x_test:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')

with open('dataset_vec/label_test.txt','w') as f:
	for item in y_test:
		for j in item:
			f.write(str(j)+' ')
		f.write('\n')

