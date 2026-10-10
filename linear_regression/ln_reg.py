import numpy as np 


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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


X ,y = make_regression(n_samples=10000,n_features=4)
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

bias,weight = ols(X_train,y_train)
# print(bias)
# print(weight)



lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = prediction(X_test,bias=bias,weight=weight)
y_pred_m = lr.predict(X_test)

mse_scratch = mean_squared_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred_m)


print(mse_scratch)
print(mse)

""" 
Model based on MSE.
For multiple times i try with different number of data i got different error rate 


Condition if features number is 1
first 1000 samples:
my model:5.0314303740275574e-29
scikit leanrn lib: 3.9788352559313564e-28


for 5000 samples:
my model:1.3723945672110116e-31
scikit leanrn lib:5.57731030997604e-29

for 5000 samples:
my model:6.269924613857262e-29
scikit leanrn lib:6.244036383915429e-29



Condition features = 4
my model:2.3822456286696985e-27
scikit leanrn lib:5.516154357684802e-26


"""