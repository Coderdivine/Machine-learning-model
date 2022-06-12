import numpy as np
import pickle
#import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self,lr=0.001,iter=1050,hidden=68):
        self.lr = lr
        self.iter = iter
        self.hidden = hidden
        self.w1 = None 
        self.b1 = None
        self.w2 = None 
        self.b2 = None
    def fit(self,x,y):
        n_samples,n_features = x.shape
        m,n=y.shape
        self.w1 = np.random.randn(self.hidden,len(x))*0.01
        self.b1 = np.zeros((self.hidden,1))    
        self.w2 = np.random.randn(1,self.hidden)*0.01
        self.b2 = np.zeros((1,1))  
        m = y.size
        for i in range(self.iter):
            if i%50 == 0:
                print("Epoch",i)
            z1 = np.dot(self.w1,x)+self.b1 
            a1 = self.hyperbolic(z1)
            z2 = np.dot(self.w2,a1)+self.b2
            a2 = self.sigmoid(z2)
            dz2 = a2 - y
            dw2 = (1/m)*np.dot(dz2,a1.T)
            db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
            dz1 = np.dot(self.w2.T,dz2)*(1-np.power(z1,2))
            dw1 = (1/m)*np.dot(dz1,x.T)
            db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)
            
            self.w1 = self.w1 - (self.lr*dw1)
            self.w2 = self.w2 - (self.lr*dw2)      
            self.b1 = self.b1 - (self.lr*db1)
            self.b2 = self.b2 - (self.lr*db2)
    def prediction(self,x):
        z1 = np.dot(self.w1,x)+self.b1
        a1 = self.hyperbolic(z1)
        z2 = np.dot(self.w2,a1)+self.b2
        a2= self.sigmoid(z2)
        a2 = [1 if i>0.5 else 0 for i in a2[0]]
        return a2
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def hyperbolic(self, z):
        return np.tanh(z)        
    def NewRoadSpeed(self,pred):
        return pred**2  
    def acc(self,p,y):
       # print(p,y)
        return (p==y).all(axis=0).mean()
 
X = np.array([
    [0.11,0.9,0.8,0.98],
    [0.12,0.89,0.73,0.99],
    [0.34,0.41,0.71,0.81],
    [0.13,0.14,0.16,0.71],
    [0.54,0.55,0.41,0.39],
    [0.11,0.41,0.43,0.23]
])
Y = np.array([
    [0],
    [1],
    [0],
    [0],
    [1],
    [0]
])
#X = np.random.randn(6,4)
#Y = np.random.randn(6,1)
data =  np.random.randint(0,3,(5000,6))
X = data[:,0:5].T
Y = data[:,5:].T
Xt = np.random.randint(0,3,(20,5)).T
print(Y)
print(X)
print(Xt)
model = NeuralNetwork()
model.fit(X,Y)
#data = np.array([[1.3,2.4,1.6,1.12]]).T
a2 = model.prediction(Xt)
print(a2)
print(model.acc(a2,Xt))
#a2 = model.NewRoadSpeed(float(a2[0]))
#lt.plot(a2,Xt)
#plt.show()