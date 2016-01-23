import tensorflow as tf
import numpy as np


def Radial_Nearest_Neighbours(data,curr_mean,delta):
    '''
    data: List of all pixel vectors [x,y,R,G,B]
    curr_mean: currently chosen mean vector [x,y,R,G,B]
    delta: Radius of interest

    Returns a list of vectors of all Nearest Neighbors
    within a Radius delta from curr_mean in the
    hyperspace of data
    '''
    tf_data=tf.constant(data)
    tf_m=tf.constant(curr_mean)
    sub=tf.sub(tf_data,tf_m)
    div=tf.div(sub,delta)
    sess = tf.Session()
    Norm = sess.run(div)
    sess.close()
    NN=[]
    for i in range(0,WIDTH*LENGTH):
        flag=0
        ### A simple barrier to reduce computation
        for j in range(0,5):
            if Norm[i][j]>=1:
                flag=1
                break
        if flag==0:
            if np.linalg.norm(Norm[i])<1:
                NN.append(data[i])
    return NN
