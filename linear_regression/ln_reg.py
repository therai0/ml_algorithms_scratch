import numpy as np 


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# implementating ols method to find weight and bias 
def ols(X,y)->list:
    try:
        X = np.asarray(X)
        y = np.asarray(y)

        # if single feature is pass
        if X.ndim == 1:
            X = X.reshape(-1,1)

        X = np.c_[np.ones(X.shape[0]),X]

        # beta = (X^T.X)^-1 . X^T . Y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y

        bias = beta[0]
        weight = beta[1:]
        return [bias,weight]

    except Exception as e:
        print(e)


# predicting the new data
def prediction(X,weight,bias)->list:
    try:
        X  = np.asarray(X)
        y_pred = []
        # for 1D data
        temp = X * weight
        for x in temp:
            y_pred.append(sum(x)+ bias)
        return y_pred
    
    except Exception as e:
        print(e)


X ,y = make_regression(n_samples=1000,n_features=1)
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

bias,weight = ols(X_train,y_train)
print(bias)
print(weight)

y_pred = prediction(X_test,bias=bias,weight=weight)
print(y_pred)