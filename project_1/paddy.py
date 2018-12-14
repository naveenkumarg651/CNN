import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
list_dir=os.listdir("G:\....\project_1")
files=glob.glob("G:\.....\project_1\*.jpg")

class paddy:
    def __init__(self):
        self.images=[]
        self.resize_image=[]
        self.classes=[1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        self.learning_rate=0.01
        self.batch_size=1
        self.epochs=100
        self.x=tf.placeholder(tf.float32,[None,150,150,3])
        self.y=tf.placeholder(tf.float32,[1])
        for i in files:
            self.image=Image.open(i)
            self.images.append(self.image)
            self.resize_image.append(np.array(self.image.resize((150,150))))
        
    def conv2d(self,x,w,bias,n,s=1):
        conv=tf.nn.conv2d(x,w,strides=[1,s,s,1],padding="VALID",name=n)
        conv=tf.nn.bias_add(conv,bias)
        conv=tf.nn.relu(conv)
        return conv
    def maxpooling(self,x,n,k=2,s=2):
        max_pool=tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,s,s,1],padding="VALID",name=n)
        return max_pool
    def convolution_net(self,x,weights):
        
        conv=self.conv2d(x,weights["w1"],weights["b1"],"layer1")
        conv=self.maxpooling(conv,"pool1")
        conv=self.conv2d(conv,weights["w2"],weights["b2"],"layer2")
        conv=self.maxpooling(conv,"pool2")
        conv=self.conv2d(conv,weights["w3"],weights["b3"],"layer3")
        conv=self.maxpooling(conv,"pool3")
        
        dense=tf.reshape(conv,[-1,weights["wd1"].get_shape().as_list()[0]])
        
        fc1=tf.sigmoid(tf.add(tf.matmul(dense,weights["wd1"]),weights["bd1"]))
        fc2=tf.sigmoid(tf.add(tf.matmul(fc1,weights["wd2"]),weights["bd2"]))
        
        return fc2
        
    def main(self):
        with tf.variable_scope("project_1",reuse=tf.AUTO_REUSE):
            weights={"w1":tf.Variable(tf.random_normal([3,3,3,32])),
            "b1":tf.Variable(tf.random_normal([32])),
            
            "w2":tf.Variable(tf.random_normal([3,3,32,64])),
            "b2":tf.Variable(tf.random_normal([64])),
            
            "w3":tf.Variable(tf.random_normal([3,3,64,64])),
            "b3":tf.Variable(tf.random_normal([64])),
            
            
            
            "wd1":tf.Variable(tf.random_normal([18496,200])),
            "bd1":tf.Variable(tf.random_normal([200])),
            
            "wd2":tf.Variable(tf.random_normal([200,1])),
            "bd2":tf.Variable(tf.random_normal([1]))}
            pred=self.convolution_net(self.x,weights)
            error=tf.square(pred-self.y)
            train=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(error)
            init=tf.initialize_all_variables()
            sess=tf.Session()
            sess.run(init)
            print("\n training.....")
            for i in range(self.epochs):
                for j in range(len(self.resize_image)):
                    prediction,cost,train1=sess.run([pred,error,train],feed_dict={self.x:[self.resize_image[j]],self.y:[self.classes[j]]})
                    if i==49:
                        print("training prediction..")
                        print(prediction)
            #testing
            
            test_image=Image.open("G:/mypython/project_1/test/7.jpg")
            test=np.float32(np.array(test_image.resize((150,150))))
            print(np.shape(test))
            output=self.convolution_net(self.x,weights)
            if sess.run([output],feed_dict={self.x:[test]})>[0.5]:
                print("diseased")
            else:
                print("not diseased")
obj=paddy()
obj.main()
                
                

  
  