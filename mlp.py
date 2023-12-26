from NNLib import *
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
class MLP:
    def __init__(self,data,batchSize):
        self.data = data
        self.nbInstances = data.shape[0]
        self.nbFeatures  = data.shape[1]
        self.batchSize = batchSize
        self.trainingSize = int(self.nbInstances*0.7)
        self.testingSize  = self.nbInstances-self.trainingSize
        self.trainingData = []
        self.testingData = []
        
        self.test_indices = np.random.permutation(self.testingSize)+100
        
        for i in range(len(self.test_indices)):
            self.testingData.append(self.data[self.test_indices[i]])
        for j in range(self.data.shape[0]):
            if j not in self.test_indices:
                self.trainingData.append(self.data[j])
        self.testingData = np.array(self.testingData)
        self.trainingData = np.array(self.trainingData)
        
        
        self.W1 = np.random.randn(5,13)*0.1
        self.b1 = np.zeros((5,1))

        self.W2 = np.random.randn(1,5)*0.1
        self.b2 = np.zeros((1,1))
    
    def loadAttributesAndLabels(self,dataSet,index,batchSize):
        i = 0
        labels = []
        if(index+batchSize > self.trainingSize):
            batchSize = batchSize-self.trainingSize%batchSize-1
        X = np.zeros((self.nbFeatures-1,batchSize))
        for j in range(index,index+batchSize):
            X[:,i] = dataSet[:,:-1][j]
            labels.append(dataSet[j][-1])
            i = i + 1
        Y = np.array([labels])
        return X,Y
    def shuffleTrainingData(self):
        random.seed(42)
        for i in range(len(self.trainingData)-1,-1,-1):
            n = random.randint(0,i)
            value = self.trainingData[i].copy()
            self.trainingData[i] = self.trainingData[n]
            self.trainingData[n] = value
    
    def forward_prop(self,input_data):
        Z1 = np.dot(self.W1,input_data) + self.b1
        A1 = tanh(Z1)
        Z2 = np.dot(self.W2,A1)+self.b2
        A2 = sigmoid(Z2)
        return Z1,A1,Z2,A2
    
    def trainingEpoch(self,inp,out):
        Z1,A1,Z2,A2 = self.forward_prop(inp)
        err = entropy(out,A2)
        m = inp.shape[1]
        Delta2 = A2 - out
        dW2 = 1.0/m * np.dot(Delta2,A1.T)
        db2 = 1.0/m * np.sum(Delta2,axis = 1,keepdims = True)
        
        Delta1 = np.dot(self.W2.T,Delta2)*tanhDeriv(Z1)
        
        dW1 = 1.0/m * np.dot(Delta1,inp.T)
        db1 = 1.0/m * np.sum(Delta1,axis = 1,keepdims = True)
        
        self.W1 = self.W1 - 0.01 * dW1
        self.b1 = self.b1 - 0.01 * db1
        
        self.W2 = self.W2 - 0.01 * dW2
        self.b2 = self.b2 - 0.01 * db2
        
        return err
    def train(self,nb_epoch):
        f = open("training.out", "w")
        f.write('#Epoch'+"\t"+'Training Error'+'\t'+'Testing Error\n')
        early_stop=0
        train_list = []
        test_list =  []
        for one_epoch in range(nb_epoch):
            self.shuffleTrainingData()
            if(one_epoch%10 ==0):
                print("[Epoch: {}]".format(one_epoch))
            index = 0
            num_batches = math.ceil(self.trainingSize/self.batchSize)
            traines = []
            for i in range(num_batches):
                X_train,Y_train = self.loadAttributesAndLabels(self.data,index,self.batchSize)
                index+=self.batchSize
                training_error = self.trainingEpoch(X_train,Y_train)
                traines.append(training_error)
            train_list.append(np.mean(traines))
            X_test = self.testingData[:,:-1].T
            Y_test = self.testingData[:,-1].reshape(1,self.testingData.shape[0])
            a1,z1,z2,a2 = self.forward_prop(X_test)
            test_list.append(entropy(Y_test,a2))
            if(one_epoch != 0) and (test_list[-1]>=test_list[-2]):
                early_stop+=1
            else:
                early_stop=0

            if(early_stop==10):
                print('Overfitting Risk. Iteration interrupted at the {} epoch'.format(one_epoch))
            

            trainingProgress = str(one_epoch)+"\t"+str(np.mean(traines))+'\t'+str(entropy(Y_test,a2))+'\n'
            f.write(trainingProgress)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Model error as a function of training epoch')
        plt.grid(color='black', ls = '-.', lw = 0.25)
        plt.plot(train_list,label='Training Error')
        plt.plot(test_list,label='Testing Error')
        plt.legend()
        plt.show()
    
    def model_evaluation(self):
        TP,TN,FP,FN = 0,0,0,0
        X_test = self.testingData[:,:-1].T
        Y_test = self.testingData[:,-1].reshape(1,self.testingData.shape[0])
        a1,z1,z2,a2 = self.forward_prop(X_test)
        for i in range(a2.shape[1]):
            if(a2[:,i]>=0.5):
                a2[:,i] = 1
            else:
                a2[:,i]=0
        count = 0
        for i in range(a2.shape[1]):
            if(a2[:,i]==1. and Y_test[:,i]==1.):
                TP += 1
            if(a2[:,i]==0. and Y_test[:,i]==0.):
                TN += 1
            if(a2[:,i]==1. and Y_test[:,i]==0.):
                FP += 1
            if(a2[:,i]==0. and Y_test[:,i]==1.):
                FN += 1
        
        precision = TP/(TP+FP)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = (TP)/(TP + FN)
        specifity = TN/(TN+FP)
        print("PRECISION : {}".format(precision))
        print("ACCURACY : {}".format(accuracy))
        
        print("SENSITIVITY : {}".format(sensitivity))
        print("SPECIFITY : {}".format(specifity))