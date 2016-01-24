from PIL import Image
import tensorflow as tf
import numpy as np


def Radial_Nearest_Neighbours(data,x,y):
    tf_data=tf.constant(data)
    mean=data[x*WIDTH+y]
    tf_m=tf.constant(m)
    sub=tf.sub(tf_data,tf_m)
    div=tf.div(sub,delta)
    sess = tf.Session()
    Norm = sess.run(div)
    sess.close()
    NN=[]
    for i in range(0,WIDTH*LENGTH):
        flag=0
        for j in range(0,5):
            if Norm[i][j]>=1:
                flag=1
                break
        if flag==0:
            if np.linalg.norm(Norm[i])<1:
                NN.append(Norm[i])
    return NN
