#coding=utf-8

#importation de la biblio numpy
import numpy as np


x_entre=np.array(([60,1.60,35],[70,1.78,40],[50,1.51,36],[52,1.56,34],[40,1.48,41],[70,1.71,42],[54,1.60,37],[80,1.83,43],[70,1.82,44]),dtype=float)
#tableau de la matrice de donnée:x_entre
y=np.array(([1],[1],[0],[0],[0],[1],[0],[1]),dtype=float) 

x_entre=x_entre/np.amax(x_entre,axis=0)

xi=np.split(x_entre,[8])[0]

xp=np.split(x_entre,[8])[1]


class neuronnes(object):
    def __init__(self):
        self.entrysynapse = 3
        self.outputsynapse = 1
        self.hiddensynapse = 6
        self.poid1=np.random.randn(self.entrysynapse,self.hiddensynapse)
        self.poid2=np.random.randn(self.hiddensynapse,self.outputsynapse)

    def forward(self,x):
        self.z=np.dot(x,self.poid1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.poid2)
        output=self.sigmoid(self.z3)
        return output
    
    def sigmoid(self,s):
        return (1/(1+np.exp(-s)))
    
    def sigmoidder(self,s):
        return s*(1-s)
    
    def backward(self,x,y,o):
        self.o_error = y - o
        self.delta = self.o_error * self.sigmoidder(o)
        
        self.z2_error=self.delta.dot(self.poid2.T)
        self.z2_delta=self.z2_error * self.sigmoidder(self.z2)
        
        self.poid1 += x.T.dot(self.z2_delta)
        self.poid2 += self.z2.T.dot(self.delta)
        
    def train(self,x,y):
        o=self.forward(x)
        self.backward(x,y,o)
    def prediction(self):
        print("résultat de la pédiction:\n")
        print("valeur entrée:\n"+str(xp)+"\n")
        print("valeur prédite:\n"+str(self.forward(xp)))
        if(self.forward(xp)<0.5):
            print("c'est du premier du genre:0")
        else:
            print("c'est du second genre:1")
        
new=neuronnes()

for i in range(10000):
    print("#"+str(i)+"\n")
    print("valeur entree:\n"+str(xi))
    print("valeur de sortie:\n"+str(y))
    print("valeur prédite par l'IA:\n"+(str(np.matrix.round(new.forward(xi),2))))
    print("\n")       
    new.train(xi,y)
new.prediction()
