#encoding=utf-8
#coding:utf-8
import jieba
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

numer = [368,308,290,123,303,351,320,214]

merge_out = open("data_text.txt","w")
label_out = open("data_label.txt","w")

for i in range(8):
    for j in range(numer[i]):
        path = ("patch/" + str(i) + "/data/" + str(j) +"/")
        f = open(path + "text.txt","r")
        lines = f.readlines()      #读取全部内容 ，并以列表方式返回  

        fout = open(path + "text_divided.txt","w")

        seg_set = ['']
        seg_out = []
        for line in lines:
	        seg_list = jieba.cut(line)
	        #print(" ".join(seg_list))
	        #fout.write(" ".join(seg_list))
	        merge_out.write(" ".join(seg_list))
	        label_out.write(str(i)+'\n')
	        print 'class:' + str(i) + ':' + str(j) +'/'+ str(numer[i])


