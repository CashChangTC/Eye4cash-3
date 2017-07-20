from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from os.path import basename
import os
import cv2
import glob
import alexnet
import numpy as np
import argparse

def get_all_image(train_dir):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    arr = []
    for file in glob.glob(train_dir + '*.*'): 
        print(file) 
        im = cv2.imread(file)
        resized_image = cv2.resize(im, (512, 512))    
        #resized_image -= int(np.mean(resized_image))
        #resized_image[resized_image<0] = 0
        resized_image = np.divide(resized_image, 128)
        arr.append(resized_image)

    return arr

def evaluate_one_image(folder):
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    train_dir = folder + '/'
    image_array = get_all_image(train_dir)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 6

        xs = tf.placeholder(tf.float32, [None, 512, 512, 3]) # 512*512*3
        keep_prob_tensor = tf.placeholder(tf.float32) #dropout
        learning_rate_tensor=tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        logit, _ = alexnet.alexnet_v2(xs, num_classes=6,spatial_squeeze=False)

        logit = tf.nn.softmax(logit)
        # you need to change the directories to yours.
        logs_train_dir = 'log/'
                       
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  #0.1 =GPU_memory usage
        sess = tf.Session(config=config) #start Session 
            
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            global_step = 9000
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

        #for testimg in image_array:
        prediction = sess.run(logit, feed_dict={xs: image_array})
        max_index = np.argmax(prediction, axis=1)
        #print(prediction)
        #print(max_index)
        res = []
        print("\n===========Prediction Results============\n")
        for result in max_index:
            if result == 0:
                res.append("1 Cent")
            elif result == 1:
                res.append("10 Cent")
            elif result == 2:
                res.append("Quarter")
            elif result == 3:
                res.append("1 Dollar")
            elif result == 4:
                res.append("20 Dollars")
            else:
                res.append("100 Dollars")
        print(res)
        print("\n=========================================\n")
        #if max_index==0:
        #    print('This is a cat with possibility %.6f' %prediction[:, 0])
        #else:
        #    print('This is a dog with possibility %.6f' %prediction[:, 1])

def parse_args():
    parser = argparse.ArgumentParser(description= ' Hackathon prediction tool ')
    parser.add_argument('-p', dest='predict', help='Input the folder path you want to predict', required=False, default='TestDataUSD', type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    evaluate_one_image(args.predict)
