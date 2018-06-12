
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

get_ipython().magic(u'matplotlib inline')


# In[10]:


train_x , train_y , test_x, test_y , classes = load_dataset()


# In[11]:


index = 25
plt.imshow(train_x[index])


# In[4]:


train_x.shape


# In[12]:


test_x.shape


# In[6]:


test_y.shape


# In[14]:


train_X = train_x.reshape(train_x.shape[0],-1).T
test_X = test_x.reshape(test_x.shape[0],-1).T


# In[15]:


train_X.shape


# In[16]:


test_X.shape


# In[17]:


train_X/=255
test_X/=255


# In[20]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[21]:


def zeros(dim):
    w = np.zeros(shape=( dim , 1))
    b=0
    
    assert(w.shape == (dim , 1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return w,b


# In[22]:


def FFandBP(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A)))
    
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = { "dw" : dw , "db" : db}
    return grads , cost


# In[23]:


def optimize(w,b,X,Y,num_iterations , learning_rate ,print_cost = False):
    costs = []
    for i in num_iterations:
        grads ,cost = FFandBP(w,b,X,Y)
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        
        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100 ==0:
            print "Cost After Iteration %i: %f" % (i,cost)
    params = {"w": w,"b":b}
    grads = {"dw":dw ,"db":db}
        
    return params , grads , costs


# In[29]:


def predict(w,b,X):
    m = X.shape[1]
    Y_pred = np.zeros((1,m))
    w =w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        Y_pred[0,i] = 1 if A[0,i] > 0.5 else 0
    assert(Y_pred.shape == (1,m))
    
    return Y_pred


# In[26]:


def model(X_train,Y_train,X_test,Y_test , num_iterations=2000,learning_rate = 0.5 , print_cost= False):
    w , b = zeros(X_train.shape[0])
    
    parameters , grads , cost = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    
    Y_pred = predict(w,b,X_test)
    Y_pred = predict(w,b,X_train)
    
    print "train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)
    print "test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[28]:


d = model(train_X , train_y ,test_X,test_y , num_iterations = 2000,learning_rate=0.005,print_cost=True)

