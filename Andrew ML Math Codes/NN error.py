import numpy as np

x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([6,12]), dtype=float)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)
x =x/np.amax(x,axis=0)
y =y/100

class NN (object):
    def __init__ (self):
        self.input=2
        self.hiddenlayer=3
        self.output=1
        self.w1=np.random.randn(self.input,self.hiddenlayer)
        self.w2=np.random.randn(self.hiddenlayer,self.output)
    def segmoid (self, s):
        return 1/(1+np.exp(-s))
    def forword (self ,x):
        self.z=np.dot(x,self.w1)
        self.a= self.segmoid(self.z)
        self.z2=np.dot(self.a,self.w2)
        o= self.segmoid(self.z2)
    def backword (self, x, y,o):
        self.o_error= (y-o)
        self.o_delta= self.o_error*self.segmoidP(o)
        self.a_error=self.o_delta.dot(self.w2.T)
        self.a_delta= self.a_error*self.semoidP(self.a)
        self.w1 +=x.T.dot(self.a_delta)
        self.w2 +=self.a.T.dotI(self.o_delta)
    def segmoidP (self,s):
        return s * (1 - s)
    def train (self ,x ,y):
        o =self.forword(x)
        self.backword(x ,y,o)
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))

for i in range(1000): # trains the NN 1,000 times
    print ("# " + str(i) + "\n")
    print ("Input (scaled): \n" + str(x))
    print ("Actual Output: \n" + str(y))
    print ("Predicted Output: \n" + str(NN.forword(x)))
    print ("Loss: \n" + str(np.mean(np.square(y - NN.forword(x)))) )# mean sum squaredloss
    print ("\n")
NN.train(x, y)
NN.saveWeights()
NN.predict()