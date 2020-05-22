# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:12:52 2020

@author: Gregor
"""


import numpy as np
import tensorflow as tf


A = tf.constant([[2, 20, 30, 3, 6],[3, 15, 32, 2, 7],[12,4,1,10,3]])
B = tf.Variable(tf.zeros((A.shape), tf.int32))

print(B)
for i in range(B.shape[0]):
    B[i, tf.argmax(A[i,:])].assign(1)
    print(B[i,:])

print(B)


# indices = tf.reshape(tf.argmax(A, axis=1), (-1,1))
# maxvals = tf.reshape(tf.reduce_max(A, axis=1), (-1,1))

# for i, row in enumerate(B):
#     row[i] = tf.reduce_max(A)
#     #tf.tensor_scatter_nd_update(row, indices[i], maxvals[i])

# B[range(len(A)), tf.argmax(A,axis=1, keepdimensions=True)] = 1

# tf.constant(B, dtype='float32')

# output = tf.zeros((1,1))
# indices = np.argmax(A, axis=1)


batch = tf.constant([[0.2,0.5, 0],[0.3,0.9,0.45],[0.25,0.4,1.2]])

m,n = batch.shape

xd_shape = [m]
c_shape = [1]
cp = batch[:,0]
print(batch)
for d in range(1,n):
    # append shape indizes
    c_shape.insert(0,m)
    xd_shape.insert(0,1)
    # get cartesian product for each dimension via broadcasting
    xd = tf.reshape(batch[:,d], (xd_shape))
    c = tf.reshape(cp,(c_shape))
    cp = tf.matmul(c , xd)

flat_cp = tf.reshape(cp,(1, m**n))
print(flat_cp)



x = batch[:,0]
y = batch[:,1]
z = batch[:,2]
A = tf.meshgrid(x, y,z)
print(A)
A = tf.meshgrid(*[row for row in tf.transpose(batch)])
print(A)

minimum = tf.reduce_min(A, axis=0)
print(tf.reshape(minimum, (1,m**n)))










@tf.function
def minimum_reasoning(signal):
    mesh = tf.meshgrid(*[row for row in tf.transpose(signal)])
    fire = tf.reduce_min(mesh, axis=0)

    return fire










CP = []
for batch in range(self.batch_size):
    signal = input_[batch,:,:]
    # produce firing strenght via min-fuzzy reasoning
    # mesh = tf.meshgrid(*[row for row in tf.transpose(signal)])
    # fire = tf.reduce_min(mesh, axis=0)
    fire = minimum_reasoning(signal)
    flat_fire = tf.reshape(fire,(1, self.m**(self.n+self.n_control)))
    CP.append(flat_fire)

return tf.reshape(tf.stack(CP), (self.batch_size, self.m**(self.n+self.n_control)))
