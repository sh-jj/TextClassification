# TextClassification
结合图片作中文文本的标签分类结合图片的文本标签分类

数据集
patch.zip

凤凰网新闻子类，每份新闻文本长度小于等于200，并爬取新闻正文中图片，正文中第一张图片用于后续工作
若新闻正文中无图片，则将凤凰网logo作为该新闻图片（default.jpg）

各类数量

0 sports 368
1 finance 308
2 tech 290
3 game 123
4 fashion 303
5 gov 350
6 culture 322
7 fo 214

共2278
6.4：1.6：2 分为训练集，验证集，测试集
数据集被随机打乱后20%作测试数据集（456）
16%作验证数据集以减少过拟合（365）

Tokenizer预处理文本数据
共35268 unique tokens
word_index = 35268
长度不足100的用0填充

cnn 作文本预测

keras 实现
TensorFlow backend.

Embedding
		in:	word_index + 1		out:	200
Dropout
		probability = 0.2
Conv1D
	filters: 250	kernel_size = 3, no padding, activation='relu', strides=1
MaxPooling1D
	kernel_size = 3, strides = kernel_size, no padding
Flatten()
Dense	(name = fc1)	
	units = 200		activation='relu'
Dense	(name = fc2)	
	units = 200		activation='relu'
Dense
		units = 8 (class_number)	activation='softmax'

Layer (type)                 Output Shape              Param #   
embedding_1 (Embedding)      (None, 100, 200)          7053800   
dropout_1 (Dropout)          (None, 100, 200)          0         
conv1d_1 (Conv1D)            (None, 98, 250)           150250    
max_pooling1d_1 (MaxPooling1 (None, 32, 250)           0         
flatten_1 (Flatten)          (None, 8000)              0         
fc1 (Dense)                  (None, 200)               1600200   
fc2 (Dense)                  (None, 200)               40200     
dense_1 (Dense)              (None, 8)                 1608      
Total params: 8,846,058

epoch = 6	batch_size = 128

测试集上	acc = 0.79
整个数据集上 acc = 0.92

lstm


Layer (type)                 Output Shape              Param #   
embedding_1 (Embedding)      (None, 100, 200)          7053800   
lstm_1 (LSTM)                (None, 200)               320800    
dropout_1 (Dropout)          (None, 200)               0         
fc1 (Dense)                  (None, 200)               40200     
dense_1 (Dense)              (None, 8)                 1608      
Total params: 7,416,408

Epoch = 10 		batch_size = 128
record
Epoch 1/10
 loss: 2.3138 - acc: 0.1606 - val_loss: 2.0209 - val_acc: 0.2438
Epoch 2/10
 loss: 1.9540 - acc: 0.2526 - val_loss: 1.9638 - val_acc: 0.2301
Epoch 3/10
 loss: 1.7626 - acc: 0.3027 - val_loss: 1.7171 - val_acc: 0.3014
Epoch 4/10
 loss: 1.6004 - acc: 0.3500 - val_loss: 1.7051 - val_acc: 0.2904
Epoch 5/10
 loss: 1.4623 - acc: 0.4688 - val_loss: 1.4187 - val_acc: 0.4849
Epoch 6/10
 loss: 1.2489 - acc: 0.5539 - val_loss: 1.3212 - val_acc: 0.5452
Epoch 7/10
 loss: 0.9688 - acc: 0.6301 - val_loss: 1.2798 - val_acc: 0.5068
Epoch 8/10
 loss: 0.8596 - acc: 0.6863 - val_loss: 1.5644 - val_acc: 0.5041
Epoch 9/10
 loss: 0.7083 - acc: 0.7687 - val_loss: 1.1237 - val_acc: 0.5890
Epoch 10/10
 loss: 0.4699 - acc: 0.8469 - val_loss: 1.2109 - val_acc: 0.5616
 
测试集
		loss			acc
[1.194813983482227, 0.5855263157894737]



特征提取
	
	图片特征提取
	
		vgg16在imagenet上训练
		使用vgg16 提取 
		http://press.liacs.nl/mirflickr/
		中images0.zip 1万张图片
		fc1层的输出(dim = 4096)
		自编码器
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
		
		optimizer='adadelta', loss='mse'
		输出10维向量作为图片特征
		
		对每篇新闻正文中的第一张图片进行特征提取得到10维向量并作正则化处理
		
	文本特征
		
		将原始新闻文本 使用 结巴中文分词	精确模式
		进行分词
		http://www.oss.io/p/fxsjy/jieba
	
		以之前cnn_model的fc1层输出作为文本特征(dim = 200)
			{测试集上	acc = 0.79
			整个数据集上 acc = 0.92}
		
	
将图片特征向量与文本特征向量连结 (dim = 210)

送mlp多层感知机

Dense	(name = fc1)	
	in : 210	out : 200	activation='relu'
Dropout
	probability = 0.2
Dense
	units = 8 (class_number)	activation='softmax'
Layer (type)                 Output Shape              Param #   
dense_1 (Dense)              (None, 200)               42200     
dropout_1 (Dropout)          (None, 200)               0         
dense_2 (Dense)              (None, 8)                 1608      
Total params: 43,808


epochs=6, batch_size=128

record

Epoch 1/6
	loss: 0.9604 - acc: 0.7687 - val_loss: 0.4710 - val_acc: 0.8740
Epoch 2/6
 loss: 0.4187 - acc: 0.9039 - val_loss: 0.3356 - val_acc: 0.9151
Epoch 3/6
 loss: 0.3207 - acc: 0.9156 - val_loss: 0.2972 - val_acc: 0.9205
Epoch 4/6
loss: 0.2833 - acc: 0.9190 - val_loss: 0.2679 - val_acc: 0.9233
Epoch 5/6
loss: 0.2665 - acc: 0.9218 - val_loss: 0.2617 - val_acc: 0.9288
Epoch 6/6
loss: 0.2586 - acc: 0.9204 - val_loss: 0.2610 - val_acc: 0.9233

测试集
		loss				acc
[0.220075376723942, 0.9276315789473685]

page2word_all.py
	将读取的新闻 使用 结巴中文分词	精确模式
	进行分词
	http://www.oss.io/p/fxsjy/jieba

	
cnn_model.py
	简单cnn模型对中文文本分类，模型及参数保存至cnn_model.h5

0.74/
0.76/
0.79/
三份cnn_model备份，测试集精度分别为0.74,0.76,0.79
	
image2vec
	将图片经vgg16、自编码器得到特征向量(dim = 10)
	输出至 vec_img.txt

text2vec.py
	将分词后的文本经cnn_model输出文本特征向量(dim = 200)
	输出至 vec_text.txt

	
vec_img.txt vec_text.txt bel.txt
为对新闻提取的图片与新闻向量

shuffle.py
	爬取的数据集类别按顺序排列
	shuffle将数据集打乱
	并分割train/test
	输出至dataset_vec/

mlp_all.py
	以提取的图片、文本特征向量作为输入
	使用多层感知机进行分类
	

	
/imgetransfer
	前期图片特征提取代码
/eztexttrans
	前期文本特征提取代码	
/dataset_vec
	打乱并划分的特征数据集
/divide
	分词测试代码
