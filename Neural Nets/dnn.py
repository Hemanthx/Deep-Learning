
# coding: utf-8

# In[4]:


import os
import numpy as np
import matplotlib.pyplot as plt
import  h5py
from testCasesv2 import *
from dnn_utils import  sigmoid , sigmoid_backward , relu , relu_backward


# In[5]:


get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

np.random.seed(1)


# In[6]:


def intialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    w1 = np.random.rand(n_h,n_x) * 0.01
    b1 = np.zeros(n_h,1)
    w2 = np.random.rand(n_y,n_h) * 0.01
    b2 = np.zeros(n_y,1)
    
        
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    
    return parameters


# In[7]:


#deep parameters
def intialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['w' + str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l],1))
        
    return parameters


# In[8]:


def linear_forward(A,W,b):
    z1 = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# In[9]:


def linear_activation_forward(A_prev , W , b ,activation):
    if activation == "sigmoid":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A, activation = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# In[10]:


def L_model_forward(X,parameters):
    caches = []
    A=X
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["w" + str(l)] , parameters["b"+str(l)],activation ="relu")
        caches.append(cache)
    AL , cache = linear_activation_forward(A_prev,parameters["w2"],parameters["b2"],activation = "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))        
    return AL, caches


# In[11]:


def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = -(1/m)*np.sum(np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


# In[12]:


#backpropgation
def linear_backward(dz,cache):
    A_prev , W,b = cache
    m = A_prev.shape[1]
    dw = (1/m)*np.dot(dz,A_prev.T)
    db = np.sum(dz,axis=1,keepdims = True)
    da_prev = np.dot(W.T,dz)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# In[13]:


def linear_activation_backward(da , cache , activation):
    linear_cache , activation_cache = cache
    if activation == "relu":
    # RELU activation function
        dz = relu_backward(da,activation_cache)
        da_prev , dw ,db = linear_backward(dz,linear_cache)
        
    elif activation == "sigmoid":
        dz = sigmoid_backward(da,activation_cache)
        da_prev , dw , db = linear_backward(dz,linear_cache)
    return da_prev , dw ,db


# In[14]:


def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y= Y.reshape(AL.shape)
    
    dal = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    ### END CODE HERE ###
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# In[15]:


def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(1,L):
        parameters["w" +  str(l+1)] = parameters["w"+str(l+1)] - learning_rate*grads["dw" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return paramters

