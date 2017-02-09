import pdb; pdb.set_trace()
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


# In[2]:

X = np.loadtxt('Viscosity.txt')
n = X.shape[0]
d = X.shape[1]
d -= 1
print(n,d)


# In[3]:

n_train = int(n*0.8)
n_test  = n - n_train
X = np.random.permutation(X)

X_train = np.zeros((n_train,d))
X_test  = np.zeros((n_test,d))
Y_train = np.zeros((n_train,1))
Y_test = np.zeros((n_test,1))

X_train[:] = X[:n_train,:-1]
Y_train[:] = np.log(X[:n_train,-1]).reshape((n_train,1))

X_test[:] = X[n_train:,:-1]
Y_test[:] = np.log(X[n_train:,-1]).reshape((n_test,1))

print(X_train.shape)
print(X_test.shape)

""" jvking normalization """
X_train = X_train - np.mean(X_train, axis = 0)
X_test = X_test - np.mean(X_test, axis = 0)
for _ in range(d):
    if np.var(X_train[:, _]) <> 0:
        X_train[:, _] = X_train[:, _] / np.std(X_train[:, _])
    if np.var(X_test[:, _]) <> 0:
        X_test[:, _] = X_test[:, _] / np.std(X_test[:, _])

# f = open('Deslist','r')
# Deslist = []
# for line in f:
#     Deslist.append(line.strip('\n\t'))
# print(Deslist)


# In[25]:

#initializing weight for first layer(w1) and second
#Parameters
hdnode = 10
w1 = np.random.normal(0,0.001,d*hdnode).reshape((d,hdnode))
d1 = np.zeros((d,hdnode))
w2 = np.random.normal(0,0.001,hdnode).reshape((hdnode,1))
d2 = np.zeros(hdnode)
h  = np.zeros(hdnode)
              
mb = 10 #minibatch size
m = int(n_train/mb)
batch = np.arange(m)
lr = 0.001 # 0.00000005
EP = 150
samp = 10000
y = np.zeros((mb,1))
yh = np.zeros((n_train,1))
yh2 = np.zeros((n_test,1))

L_train= np.zeros(EP+1)
L_test = np.zeros(EP+1)

L01_train = np.zeros((EP+1))
L01_test = np.zeros((EP+1))

#Training the neural network
def g(A):
    return (np.tanh(A))

def gd(A):
    return (1-np.square(np.tanh(A)))

for ep in range(EP):
    #print(ep)

    yh = g(X_train.dot(w1)).dot(w2)
    yh2 = g(X_test.dot(w1)).dot(w2)
        
    L_train[ep] = LA.norm(yh-Y_train)
    L_test[ep]  = LA.norm(yh2-Y_test)
    
    print(ep,L_train[ep],L_test[ep])
        
    np.random.shuffle(batch)
    for i in range(m):
        st = batch[i]*mb
        ed = (batch[i]+1)*mb
        
        h  = g(X_train[st:ed].dot(w1))
        yh = h.dot(w2)

        d2 = h.T.dot(Y_train[st:ed]-yh)
        d1 = X_train[st:ed].T.dot(np.multiply((Y_train[st:ed]-yh).dot(w2.T),gd(X_train[st:ed].dot(w1))))
        
        """ jvking gradient check """
        # for ii in range(w1.shape[0]):
        #     for jj in range(w1.shape[1]):
        #         w1[ii, jj] += 1e-7
        #         hh = g(X_train[st:ed].dot(w1))
        #         yyhh = h.dot(w2)

        w2 += lr*d2
        w1 += lr*d1


# In[26]:

yh = g(X_train.dot(w1)).dot(w2)
plt.plot(Y_train,yh,"o",color ='pink')
plt.plot(np.arange(-6,1,0.0001),np.arange(-6,1,0.0001),color = 'black')
plt.axis([-6,1,-6,1])
plt.xlabel('experiment value(log)')
plt.ylabel('prediction(log)')
plt.title('Prediction on training data')
plt.show()


# In[27]:

yh = g(X_test.dot(w1)).dot(w2)
plt.plot(Y_test,yh,"o",color ='pink')
plt.plot(np.arange(-6,1,0.0001),np.arange(-6,1,0.0001),color = 'black')
plt.axis([-6,1,-6,1])
plt.xlabel('experiment value(log)')
plt.ylabel('prediction(log)')
plt.title('prediction on test data')
plt.show()


# In[ ]:



