#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,merge
from keras import layers
from keras import layers,optimizers
import keras
#from util import CharacterTable
from sklearn.model_selection import train_test_split
import os, random
import numpy as np
from PIL import Image
import time
#import tensorflow as tf
import keras.backend as K
import string
image_h=32
image_r=200
labelmaxn=12
labelsn=26+10+1
labeln=labelsn*labelmaxn


chars = string.lowercase + string.digits + '-'
char_indices = dict((c, i) for i, c in enumerate(chars))
indice_chars = dict((i, c) for i, c in enumerate(chars)) 
#load data

def imgresize(img,image_h,image_r):
    h=img.height
    r=img.width
    n_r1=r*image_h/h
    img = img.resize((n_r1,image_h))
    rd2l = (image_r - n_r1)/2
    img = img.crop((-rd2l,0, image_r-rd2l ,image_h ))
    return img

def load_data(path):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')]
    random.seed(114)
    random.shuffle(fnames)
    data, label = [], []
    orilabel=[]
    company=[]
    for i, fname in enumerate(fnames):
        vec = fname.split('/')[-1].split('_')
        if len(vec)!=2:
            print 'errorfile:',vec
            continue
        imgLabel = vec[0]
        companyname=vec[1].split('P')[0]
        img=Image.open(fname).convert('RGB')  ####
        imgM = np.array(imgresize(img,image_h,image_r))
        
        if imgM.shape[0]*imgM.shape[1]*imgM.shape[2]!=image_h*image_r*3:
            continue
        data.append(imgM.reshape((image_h, image_r, 3)))
        lb=imgLabel.lower()
        t=[]
        for d in lb:
            t.append(char_indices[d])
        while len(t)<labelmaxn:
            t.append(-1.0)
        label.append(t)
        orilabel.append(lb)
        company.append(companyname)
    return np.array(data), np.array(label), orilabel, company

def datatransform(X_data,labels,stepsize):
    input_length=np.ones(X_data.shape[0])*stepsize
    label_length=np.sum(labels>-1,1)
    inputs = {'the_input': X_data,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length
              }
    print len(input_length),len(label_length)
    outputs = {'ctc': np.zeros([len(labels)])}  # dummy data for dummy loss function
#    return ([X_data,labels,input_length,label_length],outputs)
    return (inputs, outputs)


#set model
print 'cnn ctc model constructing'
isLoad=False
isSave=True
islog=False
model_name = 'tggvcrvpremodelv3'

#solve padding problem  Wayne----------------------
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical he re since the first couple outputs of the RNN
    # tend to be garbage:
#    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

TIMESTEP=50
input_data = Input(shape=(image_h, image_r, 3),name='the_input')
     
inner = Conv2D(128, (3, 3),padding='same',name='s1-conv1')(input_data)
inner = layers.advanced_activations.LeakyReLU(0.1,name='s1-conv1-leakyrelu')(inner)
inner = Conv2D(256, (3, 3),padding='same',name='s1-conv2')(inner)
inner = layers.BatchNormalization(name='s1-batchnorm')(inner)
inner = layers.advanced_activations.LeakyReLU(0.1,name='s1-conv2-leakyrelu')(inner)
inner = MaxPooling2D(pool_size=(2, 2),name='s1-maxpool')(inner)

inner = Conv2D(256, (3, 3),padding='same',name='s2-conv1')(inner)
inner = layers.advanced_activations.LeakyReLU(0.1,name='s2-conv1-leakyrelu')(inner)
inner = Conv2D(256, (3, 3),padding='same',name='s2-conv2')(inner)
inner = layers.BatchNormalization(name='s2-batchnorm')(inner)
inner = layers.advanced_activations.LeakyReLU(0.1,name='s2-conv2-leakyrelu')(inner)
inner = MaxPooling2D(pool_size=(2, 2),name='s2-maxpool')(inner)

inner = Conv2D(256, (3, 3),padding='same',name='s3-conv1')(inner)
#inner = Dropout(0.3,name='s3-dropout1')(inner)
inner = layers.BatchNormalization(name='s3-batchnorm')(inner)
inner = layers.advanced_activations.LeakyReLU(0.1,name='s3-conv2-leakyrelu')(inner)
inner = MaxPooling2D(pool_size=(2, 1),name='s3-maxpool')(inner)

inner = Conv2D(256, (4,1),name='s4-conv1')(inner) 
inner = Dropout(0.3,name='s3-dropout2')(inner)
inner = layers.advanced_activations.LeakyReLU(0.1,name='s4-conv1-leakyrelu')(inner)
inner = Conv2D(labelsn, (1,1),name='s4-conv2')(inner)

inner = layers.Reshape((TIMESTEP,labelsn), name='y_pred_nosoftmax')(inner)  ##may risk
y_pred = layers.Activation('softmax',name='y_pred')(inner)

labels = Input(name='the_labels', shape=[labelmaxn], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int32')
label_length = Input(name='label_length', shape=[1], dtype='int32')

loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam())

if isLoad:
    model.load_weights('models/'+model_name)

model_decode=keras.models.Model(model.get_layer('the_input').input, model.get_layer('y_pred').output)

print model.summary()


def decode(X):
    r_s=[]
    t= K.get_value(K.ctc_decode(  model_decode.predict(X),  np.ones(X.shape[0])*TIMESTEP   )[0][0])  
    for i in xrange(t.shape[0]):
        k=t[i,:].argmin()
        if t[i,k]!=-1:
            k=t.shape[1]
        r=''
        for j in xrange(k):
            r+=indice_chars[int(t[i,j])]
        r_s.append(r)
    return r_s



if islog:
    f=open('ret.txt','a')
    f.write("train on simple data set 50w,origin3. pure_cnn_v2_ctc_v2 strategy. \n")
    f.write("setting: isLoad"+str(isLoad)+'\n')
    f.write("setting: model_name"+str(model_name)+'\n')
    f.close()
print 'cnn model training'



#ninepoch=5
#epoch=int(200) #20 hour to run
#fname_list=[]
#fname_list.append('/home/bigwayne/share/HGcode/vcode/')
#fname_list.append('data/multisites/')

ninepoch=3
epoch=int(200) #20 hour to run
fname_list=[]
for i in xrange(1,11):
    name='/home/bigwayne/projects/verifycode/code2/auto/origin3_'+str(i)+'/'
    fname_list.append(name)

#ep=0
#data, label, orilabel ,company = load_data(fname_list[ep%len(fname_list)])
#x_train, x_test, y_train, y_test,label_train, label_test = train_test_split(data, label,orilabel, #test_size=0.05,random_state=791)
#x,y=datatransform(x_train,y_train,TIMESTEP)
    
starttime=time.time()
cur_epoch=0
for ep in xrange(epoch):
    # load train test data
    print 'loading train test data:',fname_list[ep%len(fname_list)]
    #data, label = load_data()
    data, label, orilabel ,company = load_data(fname_list[ep%len(fname_list)])
    x_train, x_test, y_train, y_test,label_train, label_test = train_test_split(data, label,orilabel, test_size=0.05,random_state=791)
    x,y=datatransform(x_train,y_train,TIMESTEP)
    
    print x_train.shape, y_train.shape,x_test.shape,y_test.shape
    print 'starting training...'
    model.fit(x,y, epochs=cur_epoch+ninepoch, batch_size=64,initial_epoch=cur_epoch)

    if isSave and (ep%5==0):
        timestamp=str(time.time())
        model_decode.save('models/'+model_name+'.model'+timestamp)
        model.save_weights('models/'+model_name+timestamp)
    # 测试
    nnn=500

    y_pred = decode(x_train[:nnn])

    ct=0
    for i in xrange(nnn):
        if label_train[i]==y_pred[i]:
            ct+=1
        if ct<5:
            print label_train[i],y_pred[i]
    train_acc= float(ct)/nnn

    y_pred = decode(x_test[:nnn])

    ct=0
    for i in xrange(nnn):
        if label_test[i]==y_pred[i]:
            ct+=1
        if ct < 5:
            print label_test[i],y_pred[i]
    test_acc= float(ct)/nnn
    print 'ep:',ep*ninepoch, train_acc, test_acc
    cur_epoch+=ninepoch

