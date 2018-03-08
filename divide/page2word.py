#encoding=utf-8
#coding:utf-8
import jieba
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

'''
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + " ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(" ".join(seg_list))
'''

f = open("text.txt","r")
lines = f.readlines()      #读取全部内容 ，并以列表方式返回  

fout = open("text_post.txt","w")

seg_set = ['']
seg_out = []
for line in lines:
	seg_list = jieba.cut(line)
	#print(" ".join(seg_list))
	fout.write(" ".join(seg_list))	


