
# coding: utf-8

# In[1]:


# plannar classification using neural networks
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn 
import sklearn.datasets


# In[2]:


from testCases import *
from planar_utils import plot_decision_boundary , sigmoid , load_planar_dataset,load_extra_datasets

get_ipython().magic(u'matplotlib inline')

np.random.seed(1)


# In[3]:


X,Y = load_planar_dataset()


# In[4]:


plt.scatter(X[0,:] , X[1,:])


# In[5]:


#number of training samples
X.shape


# In[6]:


Y.shape


# In[8]:


m=X.shape[1]
print m


# In[9]:


#logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)


# In[14]:


plot_decision_boundary(lambda x: clf.predict(x),X,Y)


# In[18]:


LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")


# In[19]:


# neural network model 
def layer_size(X,Y):
    n_x = X.shape[0]
    n_h =4
    n_y =Y.shape[0]
    
    return (n_x , n_h , n_y)


# In[21]:


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_size(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


# In[45]:


def intialize_parameters(n_x,n_h,n_y):
    w1 = np.random.randn(n_h,n_x) *0.01
    b1 = np.zeros((n_h,1))
    w2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    
    return parameters


# In[74]:


def foward_propagation(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    z1 = np.dot(w1,X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = np.tanh(z2)
    
    assert(a2.shape == (1, X.shape[1]))
    
    cache = {"z1": z1,
             "a1": a1,
             "z2": z2,
             "a2": a2}
    
    return a2, cache


# In[75]:


def compute_cost(a2 , Y , parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(a2),Y)
    cost = -np.sum(logprobs)
    cost = np.squeeze(cost)
    assert(isinstance(cost , float))
    
    return cost


# In[76]:


def backward_propagation(parameters , cache , X,Y):
    m = X.shape[1]
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    
    #backward propagation 
    dz2 = a2 - Y
    dw2 = (1/m) * np.dot(dz2,a1.T)
    db2 = (1/m) * np.sum(dz2,axis=1,keepdims = True)
    dz1 = np.dot(w2.T,dz2) * (1-np.power(a1,2))
    dw1 = (1/m) * np.dot(dz1,X.T)
    db1 = (1/m) * np.sum(dz1 , axis=1 , keepdims=True)
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    
    return grads


# In[77]:


def update_parameters(parameters , grads , learning_rate = 1.2):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    dw1 = grads["dw1"]
    dw2 = grads["dw2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2
    b1 = b1-learning_rate*db1
    b2 = b1-learning_rate*db2
    
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    
    return parameters


# In[78]:


def nn_model(X,Y,n_h,num_iterations = 10000,print_cost = False):
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    
    parameters = intialize_parameters(n_x,n_h,n_y)
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    for i in range(0,num_iterations):
        a2 , cache = foward_propagation(X,parameters)
        cost = compute_cost(a2,Y,parameters)
        grads = backward_propagation(parameters , cache , X , Y)
        parameters = update_parameters(parameters , grads)
        
        if print_cost and i%1000 == 0:
            print "cost after iteration %i : %f" %(i,cost)
        return parameters


# In[79]:


def predict(X , parameters):
    a2 , cache = foward_propagation(X,parameters)
    predictions = (a2 > 0.5)
    
    return predictions



# In[81]:


#build model 
parameters = nn_model(X,Y,n_h=4,num_iterations = 10000 , print_cost=True)
plot_decision_boundary(lambda x: predict(X.T,parameters), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

