import numpy as np;
data =  np.random.randint(0,3,(5000,6))
print(data)
m,n = data.shape
np.random.shuffle(data)
print(data.shape)
train_data = data[0:25]
x_train = train_data[:,0:5].T
y_train = train_data[:,5:].T
print("x_train",x_train.shape)
print("y_train",y_train.shape)
test_data = data[25:m]
x_test = test_data[:,0:5].T
y_test = test_data[:,5:].T
print(y_test)
class NeuralNetwork:
    def __int__(self,lr=0.01,iter=100,hidden=12):
        self.iter = iter
        self.hidden = hidden 
        self.lr = lr
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
    def fit(self,x,y):
        self.w1 = np.random.randn(20,len(x))*0.01
        self.b1 = np.zeros((20,1))
        self.w2 = np.random.randn(1,20)*0.01
        self.b2 = np.zeros((1,1))
        for i in range(1000000):
             z1,a1,z2,a2  =  self.forward(x,y)
             self.backward(z1,a1,z2,a2,y,x)
             if i % 10000 == 0:
                print("Epoch:",i)
                print("__accuracy__: ",i)
    def forward(self,x,y):
        z1 = np.dot(self.w1,x) + self.b1
        a1 = self.hyperbolic(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = self.sigmoid(z2)
        return z1,a1,z2,a2
    def predict(self,x):
        z1 = np.dot(self.w1,x) + self.b1
        a1 = self.hyperbolic(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = self.sigmoid(z2)
        return a2
    def backward(self,z1,a1,z2,a2,y,x):
        dz2 = a2 - y
        m = y.size
        dw2 = (1/m) * np.dot(dz2,a1.T)
        db2 = (1/m) * np.sum(dz2,axis=1,keepdims=True)
        dz1 = np.multiply(np.dot(self.w2.T,dz2),(z1*(1-z1)))
        dw1 = (1/m) * np.dot(dz1,x.T)
        db1 = (1/m) * np.sum(dz1,axis=1,keepdims=True)
        self.w2 = 0.01 * dw2
        self.b2  = 0.01 * db2
        self.w1  = 0.01 * dw1
        self.b1  = 0.01 * db1

    def hyperbolic(self, z):
        return np.tanh(z)
    def relu(self,z):
        return np.maximum(0,z)
    def sigmoid(self,z):
        return 1/(1+np.exp(z))
    def drelu(self,z):
        return z > 0
    def one_hot(self,y):
        one_hot = np.zeros((y.size,y.max()+1))
        one_hot[np.arange(y.size,y)] = 1
        one_hot = one_hot.T 
        return one_hot
    def acc(self,p,y):
       # print(p,y)
        return np.sum(p==y)/y.size


model = NeuralNetwork()
model.fit(x_train,y_train)
predict = model.predict(x_train)
def aside(x,y):
    print("length is :",len(x.T))
    for i in range(len(x.T)):
        print("values",x[0][i])
        print("predictions",y[0][i])
#print("accuracy",model.acc(predict,y_test))
aside = aside(y_train,predict)
print(aside)


