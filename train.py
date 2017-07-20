# coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.ops import init_ops
tf.logging.set_verbosity('ERROR')
from matplotlib.pyplot import imshow
from os import listdir
from os.path import basename
import os
import cv2
import glob
import alexnet
get_ipython().magic('matplotlib inline')

class Data:
    def address_collection(self, filepath):
        dirs = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
        arr = []
        for dir in dirs:
            for file in glob.glob(filepath + "/" + dir + '/*.jpeg'):
                arr.append(file)                
        return np.array(arr)
    
    def label_Initialization(self,storage_file):
        storage_label=[]
        for index in range(len(storage_file)):
            if basename(storage_file[index]).split("_")[0] == "1c":
                label = [1.0,0.0,0.0,0.0,0.0,0.0]
            elif basename(storage_file[index]).split("_")[0] == "10c":
                label = [0.0,1.0,0.0,0.0,0.0,0.0]
            elif basename(storage_file[index]).split("_")[0] == "25c":
                label = [0.0,0.0,1.0,0.0,0.0,0.0]
            elif basename(storage_file[index]).split("_")[0] == "1":
                label = [0.0,0.0,0.0,1.0,0.0,0.0]
            elif basename(storage_file[index]).split("_")[0] == "20":
                label = [0.0,0.0,0.0,0.0,1.0,0.0]
            else:
                label = [0.0,0.0,0.0,0.0,0.0,1.0]
            storage_label.append(label)
            
        storage_label=np.asarray(storage_label)
        return storage_label
    
    def image_address_to_image(self,address):
        image=[]
        for index in range(len(address)):
            im = cv2.imread(address[index])
            resized_image = cv2.resize(im, (512, 512))
            resized_image = np.divide(resized_image, 128)
            image.append(resized_image)
        return np.asarray(image)

    def init_random_data_index(self,X,y):
        if type(X) is not np.dtype:
            X=np.asarray(X)
        if type(y) is not np.dtype:
            y=np.asarray(y)
     
        batch_index = np.arange(0, len(X))
        np.random.shuffle(batch_index)
        np.random.shuffle(batch_index)
        shuf_features = X[batch_index]
        shuf_labels = y[batch_index]

        return shuf_features,shuf_labels
    
    def BatchGD_iter(self,X,y):
        batchsize=self.batchsize
        data_numbers=len(X)
        start=0
        recircle=False
        shuf_features,shuf_labels=self.init_random_data_index(X,y)
        while True:
            # shuffle labels and features
            end=start+batchsize
            
            #if next batch data out of current random index
            if end >= data_numbers:
                remainder=end-data_numbers
                end=data_numbers
                recircle=True
            
            features_batch=shuf_features[start:end]
            labels_batch=shuf_labels[start:end]
            start=end
            #if next batch data out of current random index  init a new random index
            if recircle == True:
                recircle=False
                start=0
                shuf_features,shuf_labels=self.init_random_data_index(X,y)
                features_batch=np.append(features_batch,X[start:remainder])
                labels_batch=np.concatenate([labels_batch,y[start:remainder]])
                start=remainder
            #print(features_batch)
            yield self.image_address_to_image(features_batch), labels_batch
    
    def get_batch(self, filepath, batch_size=10):
        self.filepath = filepath
        self.storage_file = self.address_collection(filepath)
        self.storage_label = self.label_Initialization(self.storage_file)
        self.batchsize = batch_size
        Batch_Generater = self.BatchGD_iter(self.storage_file, self.storage_label)
        return Batch_Generater

imagefilepath = "upup7_2/train"
datagen = Data()
batch_generate = datagen.get_batch(imagefilepath)

batch_xs, batch_ys = next(batch_generate)
imshow(batch_xs[0])
print(batch_ys)

xs = tf.placeholder(tf.float32, [None, 512, 512, 3])
ys = tf.placeholder(tf.float32, [None, 6])
keep_prob_tensor = tf.placeholder(tf.float32)
learning_rate_tensor=tf.placeholder(tf.float32)
global_step = tf.Variable(0, name='global_step', trainable=False)
define_model=False
log_path='/data3/Avery_Jupyter/log'

logit, _ = alexnet.alexnet_v2(xs, num_classes=6,spatial_squeeze=False)
#define loss
loss=slim.losses.softmax_cross_entropy(logit,ys)
#define optimizer for training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_tensor)
#define train operator
train_op = slim.learning.create_train_op(loss, optimizer,global_step)

prediction=tf.arg_max(logit,1)
accuracy=slim.metrics.accuracy(tf.arg_max(logit,1),tf.arg_max(ys,1))

#define summary for tensorboard
tf.summary.scalar('Loss',loss)
tf.summary.scalar('accuracy',accuracy)
summary_op=tf.summary.merge_all()
#get variables which need to be initialed 
variables_for_init=tf.global_variables_initializer()
    
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  #0.1 =GPU_memory usage
sess = tf.Session(config=config) #start Session 
#Session.run(require tensorflow run something  )
sess.run(variables_for_init)     #initiaal variables 
writer_master = tf.summary.FileWriter(log_path, sess.graph)   #define summary writer for tensorboard

Steps=10000
learning_rate=0.01
keep_prob=0.5
valid=1000

valimagefilepath = "upup7_2/train"
valdatagen = Data()
val_batch_generate = valdatagen.get_batch(valimagefilepath)
#define saver for save model variables from sess
Saver=tf.train.Saver()

for step in range(Steps):
    batch_xs, batch_ys = next(batch_generate)
    useless,train_loss, llo=sess.run([train_op,loss,logit], feed_dict={xs: batch_xs, ys: batch_ys, 
                                    keep_prob_tensor: keep_prob,learning_rate_tensor:learning_rate})
    if step % valid == 0:
        val_batch_xs, val_batch_ys = next(val_batch_generate)
        accuracy_,summary_=sess.run([accuracy,summary_op,], feed_dict={xs: val_batch_xs, ys: val_batch_ys, 
                                      keep_prob_tensor: keep_prob,learning_rate_tensor:learning_rate})
        #print('Logit'+str(llo))
        print ('train_loss:'+str(train_loss),'Accuracy:'+str(accuracy_),'Step:'+str(step))   

        writer_master.add_summary(summary_,step)
        model_path=os.path.join(log_path,'model_'+str(step)+'.ckpt')
        Saver.save(sess,model_path,global_step=step)
        
        writer_master.flush()




