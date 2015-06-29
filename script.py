import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import scipy.linalg as linalg
import os
import matplotlib.pyplot as plt
import pickle

path = "/home/mohit/Downloads/ML/Assignment 2"
os.chdir(path)
count= 0

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    means= np.array([])
    covmat= np.array([])
    a= y.flatten()
    print('values of y', a)
    for i in range(1,6):
        if i==1:
            m= X[a==i,:]
            means=np.mean(m,axis=0)
            
        else: 
            if i!=0 & i!=1:
                m= X[a==i,:]
                p=np.mean(m,axis=0)
                means= np.vstack([means,p])
    global_means= np.mean(X, axis=0)
    x= X-global_means
    print('shape',X.shape[0])
    covmat= np.dot(x.transpose(),x)/X.shape[0]
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    means= np.array([])
    covmat= np.array([])
    covmats=[]
    a= y.flatten()
    print('values of y', a)
    for i in range(1,6):
        if i==1:
            m= X[a==i,:]
            means=np.mean(m,axis=0)
            x= m-means
            covmat= np.dot(x.transpose(),x)/m.shape[0]
            covmats.append(covmat)
            
        else: 
            if i!=0 & i!=1:
                m= X[a==i,:]
                p=np.mean(m,axis=0)
                means= np.vstack([means,p])
                x= m-p
                covmat= np.dot(x.transpose(),x)/m.shape[0]
                covmats.append(covmat)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    y_calculated= np.array([])
    inverse= np.linalg.inv(covmat)
    determinant= np.linalg.det(covmat)
    for i in range(Xtest.shape[0]):
        px= []
        for j in range(means.shape[0]):
            p = np.array([])
            x= Xtest[i,:]
            u= means[j,:]
            subtracted= x-u
            exponential_term= np.exp(-(np.dot(np.dot(subtracted.transpose(),inverse),subtracted))/2)
            p= exponential_term/(sqrt((2*np.pi))*sqrt(determinant))
            px.append(p)
        if i==0:
            y_temp=np.argmax(px)
            y_calculated= y_temp+1 
        else:
            y_temp=np.argmax(px)
            y_calculated= np.vstack([y_calculated,y_temp+1])
    acc= np.mean((1.0*y_calculated == 1.0*ytest))
    
    
    size= (Xtest.shape[0], Xtest.shape[0])
    y_calculated1= np.zeros(size)
    
    inverse= np.linalg.inv(covmat)
    determinant= np.linalg.det(covmat)
    
    X1= np.arange(0, 16, 0.16)
    X2= np.arange(0, 16, 0.16)
    #plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','black'])
    fig= plt.figure(5)
    fig.suptitle('LDA Dicriminating Boundaries', fontsize=20)
    for i in range(X1.shape[0]):
        for k in range(X2.shape[0]):
            x= [X1[i], X2[k]]
            px= []
            for j in range(means.shape[0]):
                p = np.array([])
                #x= Xtest[i,:]
                u= means[j,:]
                subtracted= x-u
                exponential_term= np.exp(-(np.dot(np.dot(subtracted.transpose(),inverse),subtracted))/2)
                p= exponential_term/(sqrt((2*np.pi))*sqrt(determinant))
                px.append(p)
                y_temp=np.argmax(px)
            y_calculated1[i,k]= y_temp+1
            if y_calculated1[i,k]==1:
                plt.scatter(X1[i], X2[k], color='red')
            elif y_calculated1[i,k]==2:
                plt.scatter(X1[i], X2[k], color='yellow')
            elif y_calculated1[i,k]==3:
                plt.scatter(X1[i], X2[k], color='blue')
            elif y_calculated1[i,k]==4:
               plt.scatter(X1[i], X2[k], color='brown') 
            elif y_calculated1[i,k]==5:
                plt.scatter(X1[i], X2[k], color='green')
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    y_calculated= np.array([])
    for i in range(Xtest.shape[0]):
        px= []
        for j in range(means.shape[0]):
            p = np.array([])
            x= Xtest[i,:]
            u= means[j,:]
            subtracted= x-u
            inverse= np.linalg.inv(covmats[j])
            determinant= np.linalg.det(covmats[j])
            exponential_term= np.exp(-(np.dot(np.dot(subtracted.transpose(),inverse),subtracted))/2)
            p= exponential_term/(sqrt((2*np.pi))*sqrt(determinant))
            px.append(p)
        if i==0:
            y_temp=np.argmax(px)
            y_calculated= y_temp+1 
        else:
            y_temp=np.argmax(px)
            y_calculated= np.vstack([y_calculated,y_temp+1])
    acc= np.mean((1.0*y_calculated == 1.0*ytest))
    
    size= (Xtest.shape[0], Xtest.shape[0])
    y_calculated1= np.zeros(size)
    X1= np.arange(0, 16, 0.16)
    X2= np.arange(0, 16, 0.16)
    #plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','black'])
    fig =plt.figure(1)
    fig.suptitle('QDA Dicriminating Boundaries', fontsize=20)
    for i in range(X1.shape[0]):
        for k in range(X2.shape[0]):
            x= [X1[i], X2[k]]
            px= []
            for j in range(means.shape[0]):
                p = np.array([])
                #x= Xtest[i,:]
                u= means[j,:]
                subtracted= x-u
                inverse= np.linalg.inv(covmats[j])
                determinant= np.linalg.det(covmats[j])
                exponential_term= np.exp(-(np.dot(np.dot(subtracted.transpose(),inverse),subtracted))/2)
                p= exponential_term/(sqrt((2*np.pi))*sqrt(determinant))
                px.append(p)
            #if i==0:
                #y_temp=np.argmax(px)
                #y_calculated= y_temp+1 
           # else:
                y_temp=np.argmax(px)
            y_calculated1[i,k]= y_temp+1
            if y_calculated1[i,k]==1:
                plt.scatter(X1[i], X2[k], color='red')
            elif y_calculated1[i,k]==2:
                plt.scatter(X1[i], X2[k], color='yellow')
            elif y_calculated1[i,k]==3:
                plt.scatter(X1[i], X2[k], color='blue')
            elif y_calculated1[i,k]==4:
               plt.scatter(X1[i], X2[k], color='brown') 
            elif y_calculated1[i,k]==5:
                plt.scatter(X1[i], X2[k], color='green')
    return acc


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                  
    w = np.dot(np.dot(linalg.inv(np.dot(X.T,X)),X.T),y)                                               
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
 
    w = np.dot(np.dot(linalg.inv(np.dot(X.T,X)+np.dot(X.shape[0],np.dot(lambd,np.identity(X.shape[1])))),X.T),y)                                                                                              
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    rmse = np.divide(np.sqrt(np.mean(np.square(ytest - np.dot(Xtest, w)))*Xtest.shape[0]),Xtest.shape[0])
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    w_normal= np.reshape(w, (w.shape[0],1))
    y_transpose= np.transpose(y)
    X_transpose= np.transpose(X)
    w_transpose= np.transpose(w)
    w_X= np.dot(X, w_normal)
    first_term= np.dot(y_transpose, y)
    second_term= np.dot(y_transpose, w_X)
    third_term= np.dot(np.transpose(w_X), y)
    fourth_term= np.dot(np.transpose(w_X), w_X)
    
    error_first_term= np.divide((first_term-second_term-third_term+fourth_term), 2*X.shape[0])
    error_second_term= np.divide((np.multiply(lambd, np.dot(w,w_transpose))),2)
    
    #w_X_reshaped= np.reshape(w_X, (w_X.shape[0], 1))
    #substracted_value= y-w_X_reshaped
    #substracted_value_transpose= substracted_value.transpose()
    #value= np.dot(substracted_value_transpose,substracted_value)
    error= error_first_term+error_second_term
    
    
   
    initial_term= np.dot(y_transpose, X)
    error_grad_second_term= np.dot(w_transpose,np.dot(X_transpose, X))
    error_grad_temp= np.divide((error_grad_second_term - initial_term), X.shape[0])
    error_grad_temp2= error_grad_temp + lambd*w_transpose
    error_grad= np.squeeze(error_grad_temp2)                                                                                
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         

    p_matrix = np.zeros((x.shape[0],p+1))
    for i in range (0,p_matrix.shape[0]):
        for j in range (0, p_matrix.shape[1]):
            p_matrix[i][j] = np.power(x[i],j)
    Xd = p_matrix
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
print ('minimum value for OLE', np.min(w))
print ('maximum value for OLE', np.max(w))
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses4 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses4[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
print ('minimum value for Ridge Regression', np.min(w_l))
print ('maximum value for Ridge regression', np.max(w_l))
fig = plt.figure(2)
fig.suptitle('Ridge Regression', fontsize=20)
plt.plot(lambdas,rmses3)
plt.plot(lambdas,rmses4)
plt.legend(('Test Data','Training Data'))


# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses5 = np.zeros((k,1))
rmses6 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses5[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses6[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1
fig1= plt.figure(3)
fig1.suptitle('Ridge Regression with Gradient Descent', fontsize=20)
plt.plot(lambdas,rmses5)
plt.plot(lambdas,rmses6)
plt.legend(('Test Data','Training Data'))

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses5)]
rmses7 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses7[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses7[p,1] = testOLERegression(w_d2,Xdtest,ytest)
fig2 =plt.figure(4)
fig2.suptitle('Non Linear Ridge Regression', fontsize=20)
plt.plot(range(pmax),rmses7)
plt.legend(('No Regularization','Regularization'))
plt.show()
