import numpy as np 
#intialise inputs

Input = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
#define target vairable

output =np.array([[0],[1],[1],[0]]) 

#generate random weights 
wt1 = 2*np.random.random((3,4)) -1
wt2 = 2*np.random.random((4,1)) -1

#define non linear activation function - sigmoid
def sigmoid(x,deriv=False):
	if(deriv==True):
		return x*(1-x)

	return 1/(1-np.exp(-x))

#iterate over input many times
#forward prop and back prop

for j in xrange(60000):
	i1 = Input;
	i2 = sigmoid(np.dot(i1,wt1))
	i3 = sigmoid(np.dot(i2,wt2))

	#calculate error generated
	i3_error = output - i3
	if(j%10000) ==0:
		print "Error:" + str(np.mean(np.abs(i3_error)))

	#backpropagation change weights value
	i3_delta = i3_error*sigmoid(i3,deriv=True)
	#calculate error at layer number 2
	i2_error = i3_delta.dot(wt2.T)

	i2_delta = i2_error*sigmoid(i2,deriv=True)

	wt2= wt2 + i2.T.dot(i3_delta)
	wt1 = wt1 + i1.T.dot(i2_delta)


