
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

import PIL
from PIL import Image, ImageFile

import os
import urllib2
import cStringIO
import random

import threading 
import time 


def read_img(directory, data_type, one_hot=False, multi_channel=False):
    images = []
    labels = []
    img_name = []
    tmp = []
    
    cnt =0 

    for file_name in os.listdir(directory):
         
        label = int(file_name.split('_')[0])
        file_type = file_name.split('_')[1]
        if file_type != data_type:
            continue
        
        img = Image.open(directory + '/' + file_name)
        if not multi_channel:
            img = img.convert('L')
            img = numpy.asarray(img, dtype='uint8')
            img = img.reshape(img.shape[0], img.shape[1],1)
        else:
            img = numpy.asarray(img, dtype='uint8')
        if len(img.shape) != 3:
            print('img shape error:' + str(len(img.shape)))
            continue 
        if one_hot:
            label_list = [0, 0]
            label_list[label%2]=1
            if data_type == 'train':
                tmp.append((img[0:200,0:200,:], label_list, random.random(), file_name))
                tmp.append((img[0:200,20:220,:], label_list, random.random(), file_name))
                tmp.append((img[20:220,0:200,:], label_list, random.random(), file_name))
                tmp.append((img[20:220,20:220,:], label_list, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label_list, random.random(), file_name))
            else:
                tmp.append((img[10:210,10:210,:], label_list, random.random(), file_name))
                
        else:
            
            if data_type == 'train': 
                #tmp.append((img[0:200,0:200,:], label, random.random(), file_name))
                #tmp.append((img[0:200,20:220,:], label, random.random(), file_name))
                #tmp.append((img[20:220,0:200,:], label, random.random(), file_name))
                #tmp.append((img[20:220,20:220,:], label, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label, random.random(), file_name))
                cnt +=1
                if cnt>5000:
                    break
                if cnt%100==0:
                    print(str(cnt))
            else:
                #tmp.append((img[0:200,0:200,:], label, random.random(), file_name))
                #tmp.append((img[0:200,20:220,:], label, random.random(), file_name))
                #tmp.append((img[20:220,0:200,:], label, random.random(), file_name))
                #tmp.append((img[20:220,20:220,:], label, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label, random.random(), file_name))
                cnt +=1
                if cnt>2000:
                    break
                if cnt%100==0:
                    print(str(cnt))
                 
    tmp = sorted(tmp, cmp=lambda x,y:cmp(x[2],y[2]))
    for i in range(len(tmp)):
        images.append(tmp[i][0])  
        labels.append(tmp[i][1])  
        img_name.append(tmp[i][3])  
    return (numpy.array(images), numpy.array(labels), numpy.array(img_name))


def read_img_batch(directory, data_type, one_hot=False, multi_channel=False, offset=0, batch_size=64):
    images = []
    labels = []
    img_name = []
    tmp = []
    
    cnt =-1 

    for file_name in os.listdir(directory):
         
        label = int(file_name.split('_')[0])
        file_type = file_name.split('_')[1]
        if file_type != data_type:
            continue
        
        cnt += 1
        if cnt<offset:
            continue
        if cnt >= offset+batch_size:
            break


        img = Image.open(directory + '/' + file_name)
        
        if not multi_channel:
            img = img.convert('L')
            img = numpy.asarray(img, dtype='float32')
            img = img.reshape(img.shape[0], img.shape[1],1)
        else:
            img = numpy.asarray(img, dtype='float32')
        if len(img.shape) != 3:
            print('img shape error:' + str(len(img.shape)))
            continue 
        if one_hot:
            label_list = [0, 0]
            label_list[label%2]=1
            if data_type == 'train':
                tmp.append((img[0:200,0:200,:], label_list, random.random(), file_name))
                tmp.append((img[0:200,20:220,:], label_list, random.random(), file_name))
                tmp.append((img[20:220,0:200,:], label_list, random.random(), file_name))
                tmp.append((img[20:220,20:220,:], label_list, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label_list, random.random(), file_name))
            else:
                tmp.append((img[10:210,10:210,:], label_list, random.random(), file_name))
                
        else:
            
            if data_type == 'train': 
                tmp.append((img[0:200,0:200,:], label, random.random(), file_name))
                tmp.append((img[0:200,20:220,:], label, random.random(), file_name))
                tmp.append((img[20:220,0:200,:], label, random.random(), file_name))
                tmp.append((img[20:220,20:220,:], label, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label, random.random(), file_name))
            else:
                tmp.append((img[0:200,0:200,:], label, random.random(), file_name))
                tmp.append((img[0:200,20:220,:], label, random.random(), file_name))
                tmp.append((img[20:220,0:200,:], label, random.random(), file_name))
                tmp.append((img[20:220,20:220,:], label, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label, random.random(), file_name))
                 
    tmp = sorted(tmp, cmp=lambda x,y:cmp(x[2],y[2]))
    for i in range(len(tmp)):
        images.append(tmp[i][0])  
        labels.append(tmp[i][1])  
        img_name.append(tmp[i][3])  
    return (numpy.array(images), numpy.array(labels), numpy.array(img_name))
def read_img_online(directory, data_type, one_hot=False, multi_channel=False):
    images = []
    labels = []
    img_name = []
    tmp = []

    query_label = {}
    label_index = 0
    wid_index = -1

    INPUT = 'query_sku_img'
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    for line in open(INPUT):
        line = line.strip().split('\t')
        query = line[0]
        wid = int(float(line[1]))
        label = 0
        if query in query_label:
            label = query_label[query]
            wid_index += 1
        else:
            label = label_index
            query_label[query] = label
            label_index += 1
            wid_index = 0
    
        if (data_type == 'train' and wid_index % 10 < 2) or (data_type == 'valid' and wid_index % 10 == 7) or (data_type == 'test' and wid_index % 10 >= 9) :
            try:
                url = 'http://img13.360buyimg.com/n7/' + line[4]
                file = cStringIO.StringIO(urllib2.urlopen(url).read())
                img = Image.open(file)
            except:
                continue
            file_name = str(label) + '_' + data_type + '_' + str(wid_index) + '_' + str(wid) + '.jpg'
            img.save(directory+ '/' + file_name)

            if not multi_channel:
                img = img.convert('L')
                img = numpy.asarray(img, dtype='float32')
                img = img.reshape(img.shape[0], img.shape[1],1)
            else:
                img = numpy.asarray(img, dtype='float32')
            if len(img.shape) != 3:
                print('img shape error:' + str(len(img.shape)))
                continue
            if img.shape[2] != 3:
                print('img channel error:' + str(img.shape[2]))
                continue 
            if one_hot:
                label_list = [0, 0]
                label_list[label%2]=1
                #tmp.append((img[:,:,0], label_list, random.random()))
                tmp.append((img[0:200,0:200,:], label_list, random.random(), file_name))
                tmp.append((img[0:200,20:220,:], label_list, random.random(), file_name))
                tmp.append((img[20:220,0:200,:], label_list, random.random(), file_name))
                tmp.append((img[20:220,20:220,:], label_list, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label_list, random.random(), file_name))
            else:
                tmp.append((img[0:200,0:200,:], label, random.random(), file_name))
                tmp.append((img[0:200,20:220,:], label, random.random(), file_name))
                tmp.append((img[20:220,0:200,:], label, random.random(), file_name))
                tmp.append((img[20:220,20:220,:], label, random.random(), file_name))
                tmp.append((img[10:210,10:210,:], label, random.random(), file_name))
    tmp = sorted(tmp, cmp=lambda x,y:cmp(x[2],y[2]))
    for i in range(len(tmp)):
        images.append(tmp[i][0])  
        labels.append(tmp[i][1])  
        img_name.append(tmp[i][3])  
    return (numpy.array(images), numpy.array(labels), numpy.array(img_name))


######################################################################################################
## read img for model training

def read_img_all(directory, data_type, one_hot=False, multi_channel=False, img_num=10000):
    images = []
    labels = []
    img_name = []
    tmp = []
    
    cnt =0 

    for file_name in os.listdir(directory):
         
        label = int(file_name.split('_')[0])
        #file_type = file_name.split('_')[1]
        #if file_type != data_type:
        #    continue
        
        img = Image.open(directory + '/' + file_name)
        if not multi_channel:
            img = img.convert('L')
            img = numpy.asarray(img, dtype='uint8')
            img = img.reshape(img.shape[0], img.shape[1],1)
        else:
            img = numpy.asarray(img, dtype='uint8')
        if len(img.shape) != 3:
            print('img shape error: the img ' + file_name + ' has an error shape of ' + str(len(img.shape)))
            continue 
        if one_hot:
            label_list = [0, 0]
            label_list[label%2]=1
            tmp.append((img, label_list, random.random(), file_name))
                
        else:
            tmp.append((img, label, random.random(), file_name))
            cnt +=1
            if cnt>img_num:
                break
            if cnt%1000==0:
                print('already loaded: ' + str(cnt) + ' imgs')
                 
    tmp = sorted(tmp, cmp=lambda x,y:cmp(x[2],y[2]))
    for i in range(len(tmp)):
        images.append(tmp[i][0])  
        labels.append(tmp[i][1])  
        img_name.append(tmp[i][3])  
    return (numpy.array(images), numpy.array(labels), numpy.array(img_name))


class read_img_for_train_base_thread(threading.Thread):  #继承父类threading.Thread 
    
    def __init__(self, name, directory, img_list, one_hot, multi_channel, flip_img, sub_mean): 
        threading.Thread.__init__(self) 
        self.name = name 
        self.directory = directory
        self.img_list = img_list
        self.one_hot = one_hot
        self.multi_channel = multi_channel
        self.flip_img = flip_img
        self.sub_mean = sub_mean
        self.ret_list = []
    
    def run(self):  #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        
        print('Starting ' + self.name + ': ' + str(len(self.img_list)) + 'imgs to read')  
        cnt = 0
        

        if self.flip_img:
            print('horizonal flip img works')
        for file_name in self.img_list:
            label = int(file_name.split('_')[0])
            wid = file_name.split('_')[1].split('.')[0]
            try:
                img = Image.open(self.directory + '/' + file_name)
            except:
                print('img open error:' + file_name)
                continue
            if not self.multi_channel:
                img = img.convert('L')
                img_np = numpy.asarray(img, dtype='uint8')
                img_np = img_np.reshape(img_np.shape[0], img_np.shape[1],1)
            else:
                img_np = numpy.asarray(img, dtype='uint8')
            
                if len(img_np.shape) != 3:
                    print('img shape error:' + str(len(img_np.shape)) + ', convert to RGB')
                    img = img.convert('RGB')
                    img_np = numpy.asarray(img, dtype='uint8')
                if self.sub_mean:
                    print('img sub mean works')
                    img_np = numpy.subtract(img_np, [197.4186934, 190.77458879, 186.91008562])

            if self.one_hot:
                label_list = [0, 0]
                label_list[label%2]=1
                self.ret_list.append((img_np, label_list, random.random(), file_name))
                if self.flip_img:
                    self.ret_list.append((img_np[:,::-1,:], label_list, random.random(), file_name))
                 
            else:
                self.ret_list.append((img_np, label, random.random(), file_name))
                if self.flip_img:
                    self.ret_list.append((img_np[:,::-1,:], label, random.random(), file_name))
        
            cnt +=1
            if self.flip_img:
                cnt +=1
                
            if cnt%1000==0:
                print('threading ' + self.name + ': img loaded ' + str(cnt))
        
        print('Ending ' + self.name + ': ' + str(cnt) + 'imgs loaded') 
        

def read_img_for_train_multi_thread(directory, one_hot=False, multi_channel=False, THREAD_NUM=10, img_num=sys.maxint, flip_img=False, sub_mean=False):
    
    images = []
    labels = []
    img_name = []
    
    img_list = []
    cnt = 0
    for file_name in os.listdir(directory):
        img_list.append(file_name) 
        cnt += 1
        if cnt >= img_num:
            break
    print(str(len(img_list)) + ' imgs to load in total, use ' + str(THREAD_NUM) + ' threadings')
    
    each_thread_num = len(img_list)//THREAD_NUM
    _thread_list = []
    
    cnt = 0
    for idx in range(0, len(img_list), each_thread_num):
        # 创建新线程 
        _thread = read_img_for_train_base_thread("Thread-" + str(cnt), directory, img_list[idx:idx+each_thread_num], one_hot, multi_channel, flip_img, sub_mean) 
        _thread_list.append(_thread)
        cnt +=1

    for _thread in _thread_list:
        # 开启线程 
        _thread.start() 
    
    _thread_ret = []
    for _thread in _thread_list:
        _thread.join()
        _thread_ret += _thread.ret_list
    
    print('read threading done')
    
    _thread_ret = sorted(_thread_ret, cmp=lambda x,y:cmp(x[2],y[2]))
    
    for i in range(len(_thread_ret)):
        images.append(_thread_ret[i][0])  
        labels.append(_thread_ret[i][1])  
        img_name.append(_thread_ret[i][3])  
    
    return (images, labels, img_name)
    
        
                 
##############################################################################################
## read img for model predicting
def read_img_for_test(directory, multi_channel=False, img_num=10000):
    images = []
    img_name = []
    
    cnt =0 

    for file_name in os.listdir(directory):
        wid = file_name.split('.')[0]
        try:
            img = Image.open(directory + '/' + file_name)
        except:
            print('img open error:' + file_name)
            continue
        if not multi_channel:
            img = img.convert('L')
            img_np = numpy.asarray(img, dtype='uint8')
            img_np = img_np.reshape(img_np.shape[0], img_np.shape[1],1)
        else:
            
            img_np = numpy.asarray(img, dtype='uint8')
            
            if len(img_np.shape) != 3:
                print('img shape error:' + str(len(img_np.shape)) + ', convert to RGB')
                img = img.convert('RGB')
                img_np = numpy.asarray(img, dtype='uint8')
                
        #images.append(img[10:210,10:210,:])
        images.append(img_np)
        img_name.append(wid) 
        
        cnt +=1
        if cnt>img_num:
            break
        if cnt%10000==0:
            print(str(cnt))
                 
    return (images, img_name)
    
class read_img_for_test_base_thread(threading.Thread):  #继承父类threading.Thread 
    
    def __init__(self, name, directory, img_list, multi_channel): 
        threading.Thread.__init__(self) 
        self.name = name 
        self.directory = directory
        self.img_list = img_list
        self.multi_channel = multi_channel
        self.ret_list = []
    
    def run(self):  #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        
        print('Starting ' + self.name + ': ' + str(len(self.img_list)) + 'imgs to read')  
        cnt = 0
    
        for file_name in self.img_list:
            wid = file_name.split('.')[0]
            try:
                img = Image.open(self.directory + '/' + file_name)
            except:
                print('img open error:' + file_name)
                continue
            if not self.multi_channel:
                img = img.convert('L')
                img_np = numpy.asarray(img, dtype='uint8')
                img_np = img_np.reshape(img_np.shape[0], img_np.shape[1],1)
            else:
                img_np = numpy.asarray(img, dtype='uint8')
            
                if len(img_np.shape) != 3:
                    print('img shape error:' + str(len(img_np.shape)) + ', convert to RGB')
                    img = img.convert('RGB')
                    img_np = numpy.asarray(img, dtype='uint8')
                
            self.ret_list.append((img_np, wid))
        
            cnt +=1
            if cnt%1000==0:
                print('threading ' + self.name + ': img loaded ' + str(cnt))
        
        print('Ending ' + self.name + ': ' + str(cnt) + 'imgs crawled') 

def read_img_for_test_multi_thread(directory, multi_channel=False, THREAD_NUM=10, img_num=sys.maxint):
    
    images = []
    img_name = []
    
    img_list = []
    cnt = 0
    for file_name in os.listdir(directory):
        img_list.append(file_name) 
        cnt += 1
        if cnt >= img_num:
            break

    print(str(len(img_list)) + ' imgs to load in total, use ' + str(THREAD_NUM) + ' threadings')
    
    each_thread_num = len(img_list)//THREAD_NUM
    _thread_list = []
    
    cnt = 0
    for idx in range(0, len(img_list), each_thread_num):
        # 创建新线程 
        _thread = read_img_for_test_base_thread("Thread-" + str(cnt), directory, img_list[idx:idx+each_thread_num], multi_channel) 
        _thread_list.append(_thread)
        cnt +=1

    for _thread in _thread_list:
        # 开启线程 
        _thread.start() 
    
    _thread_ret = []
    for _thread in _thread_list:
        _thread.join()
        _thread_ret += _thread.ret_list
    
    print('read threading done')
    
    for img, na in _thread_ret:
        images.append(img)
        img_name.append(na)
    
    return (images, img_name)

########################################################################################################
# crawl img for train
class crawl_img_for_train_base_thread(threading.Thread):  #继承父类threading.Thread 
    
    def __init__(self, name, directory, img_list): 
        threading.Thread.__init__(self) 
        self.name = name 
        self.directory = directory
        self.img_list = img_list 
    
    def run(self):  #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        
        print('Starting ' + self.name + ': ' + str(len(self.img_list)) + 'imgs to crawl')  
        
        cnt = 0
        for wid, img_url, label in self.img_list:
            try:
                url = 'http://img13.360buyimg.com/n7/' + img_url
                file = cStringIO.StringIO(urllib2.urlopen(url).read())
                img = Image.open(file)
                file_name = str(label) + '_' + str(wid) + '.jpg'
                img.save(self.directory+ '/' + file_name)
                cnt += 1
            except:
                print(self.name + ': img open and save error:' + str(wid))
                continue
         
        print('Ending ' + self.name + ': ' + str(cnt) + 'imgs crawled') 


def crawl_img_for_train_multi_thread(directory, INPUT, THREAD_NUM=10, query_idx=0, wid_idx=1, url_idx=4, field_len=5):
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    query_label = {}
    label_index = 0
    img_list = []
    for line in open(INPUT):
        line = line.strip().split('\t')
        if len(line)<field_len:
            print(line[0] + ' img file field error, the field len is ' + str(len(line)))
            continue
        query = line[query_idx]
        wid = int(float(line[wid_idx]))
        url = line[url_idx]
        
        label = 0
        if query in query_label:
            label = query_label[query]
        else:
            label = label_index
            query_label[query] = label
            label_index += 1
        
        img_list.append((wid, url, label))
    for k in query_label.keys():
        print(k + '\t' + str(query_label[k]))
    ''' 
    print(str(len(img_list)) + ' imgs to crawl in total, use ' + str(THREAD_NUM) + ' threadings')
    each_thread_num = len(img_list)//THREAD_NUM

    for idx in range(0, len(img_list), each_thread_num):
         
        # 创建新线程 
        _thread = crawl_img_for_train_base_thread("Thread-" + str(idx), directory, img_list[idx:idx+each_thread_num]) 

        # 开启线程 
        _thread.start() 

    print('crawl threading done')
    '''
########################################################################################################
## crawl img for test 
## 
def crawl_img_for_test(directory, INPUT, wid_idx=0, url_idx=1):
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    for line in open(INPUT):
        line = line.strip().split('\t')
        wid = int(float(line[wid_idx]))
        try:
            url = 'http://img13.360buyimg.com/n7/' + line[url_idx]
            file = cStringIO.StringIO(urllib2.urlopen(url).read())
            img = Image.open(file)
            file_name = str(wid) + '.jpg'
            img.save(directory+ '/' + file_name)
        except:
            print('img error:' + str(wid))
            continue

    return



class crawl_img_for_test_base_thread(threading.Thread):  #继承父类threading.Thread 
    
    def __init__(self, name, directory, img_list): 
        threading.Thread.__init__(self) 
        self.name = name 
        self.directory = directory
        self.img_list = img_list 
    
    def run(self):  #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        
        print('Starting ' + self.name + ': ' + str(len(self.img_list)) + 'imgs to crawl')  
        
        cnt = 0
        for wid, img_url in self.img_list:
            try:
                url = 'http://img13.360buyimg.com/n7/' + img_url
                file = cStringIO.StringIO(urllib2.urlopen(url).read())
                img = Image.open(file)
                file_name = str(wid) + '.jpg'
                img.save(self.directory+ '/' + file_name)
                cnt += 1
            except:
                print(self.name + ': img open and save error:' + str(wid))
                continue
         
        print('Ending ' + self.name + ': ' + str(cnt) + 'imgs crawled') 


def crawl_img_for_test_multi_thread(directory, INPUT, THREAD_NUM=10, wid_idx=0, url_idx=1):
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    img_list = []
    for line in open(INPUT):
        line = line.strip().split('\t')
        if len(line)<2:
            print(line[0] + ' img file field error, the field len is ' + str(len(line)))
            continue
        wid = int(float(line[wid_idx]))
        url = line[url_idx]
        
        img_list.append((wid, url))
   
    print(str(len(img_list)) + ' imgs to crawl in total, use ' + str(THREAD_NUM) + ' threadings')
    each_thread_num = len(img_list)//THREAD_NUM

    for idx in range(0, len(img_list), each_thread_num):
         
        # 创建新线程 
        _thread = crawl_img_for_test_base_thread("Thread-" + str(idx), directory, img_list[idx:idx+each_thread_num]) 

        # 开启线程 
        _thread.start() 
    

    print('crawl threading done')
     

def read_img_specified(directory, img_file, multi_channel=False):
        
    ret_list = []
    ret_name = []
    cnt = 0
    for file_name in open(img_file):
        file_name = file_name.strip()
        try:
            img = Image.open(directory + '/' + file_name)
        except:
            print('img open error:' + file_name)
            continue
        if not multi_channel:
            img = img.convert('L')
            img_np = numpy.asarray(img, dtype='uint8')
            img_np = img_np.reshape(img_np.shape[0], img_np.shape[1],1)
        else:
            img_np = numpy.asarray(img, dtype='uint8')
            
            if len(img_np.shape) != 3:
                print('img shape error:' + str(len(img_np.shape)) + ', convert to RGB')
                img = img.convert('RGB')
                img_np = numpy.asarray(img, dtype='uint8')
                
        ret_list.append(img_np)
        ret_name.append(file_name)
        
        cnt +=1
        if cnt%1000==0:
            print('img loaded ' + str(cnt))
       
    return ret_list, ret_name

if __name__ == '__main__':
    img_file= './img/query_sku_img_all_cid_filter'
    img_saved_dir = './img/query_sku_img_all_cid_filter_dir'
    if not os.path.exists(img_saved_dir):
        os.mkdir(img_saved_dir)
     
    #crawl_img_for_test('./img/wid_img_edit_sample_dir' ,'./img/wid_img_edit_sample', wid_idx=0, url_idx=1)
    #crawl_img_for_test_multi_thread(img_saved_dir , img_file, THREAD_NUM=20, wid_idx=0, url_idx=1)
    crawl_img_for_train_multi_thread(img_saved_dir, img_file, THREAD_NUM=30, query_idx=0, wid_idx=1, url_idx=4, field_len=5)
        


