import numpy as np 


# implementing the OLS method to find the weight and bias value 
def ols(X,y)->list:
    try:
        # if the X value is 1D or single features
        y_mean = np.mean(y)
        

        if X.ndim == 1:
            x_mean = np.mean(X)
            temp  = []
            for i in range(len(X)):
                final_val = 0
                x_var = X[i] - x_mean
                y_var = y[i] - y_mean
                x_var_sqr = np.power(x_var,2)
                if x_var == 0:
                    temp.append(final_val)
                else:
                    final_val = (x_var * y_var) / x_var_sqr
                    temp.append(final_val)

            beta1 = sum(temp)
            beta0 = y_mean - beta1 * x_mean 
            print(beta0)
            print(beta1)
            return [beta1,beta0]

       
        x_mean = []
        beta1 = []
        beta0 = 0


        # finding the mean of each columns
        for i in range(X.shape[1]):
            x = []
            for j in range(X.shape[0]):
                x.append(X[j][i])
            x_mean.append(np.mean(x))
        
        # finding the beta value for each columns 
        for i in range(X.shape[1]):
            temp = []
            for j in range(X.shape[0]):
                final_val = 0
                x_var = X[j][i] - x_mean[i]
                y_var = y[j] - y_mean
                x_var_sqr = np.power(x_var,2)

                if x_var == 0:
                    temp.append(final_val)
                else:
                    final_val = (x_var*y_var) / x_var_sqr
                    temp.append(final_val)
            beta1.append(sum(temp))
        
        beta0 = y_mean - (sum(np.array(beta1)*np.array(x_mean)))
        return [beta1,beta0]
    except Exception as e:
        print(e)



# prediction new data 
def model_prediction(X,weight,bias):
    try:
        y_pred = []
        # if 1D data is pass
        if X.ndim == 1:
            for i in range(len(X)):
                y_pred = weight * X[i] + bias 
            y_pred.append(y_pred)
            return y_pred
         
        # for 2d data 
        else:
            for j in range(X.shape[0]):
                temp = []
                for i in range(X.shape[1]):
                    temp.append(X[j][i] * weight[i])
                y_pred.append(sum(temp)+bias)
                return y_pred
    
    except Exception as e:
        print(e)

# y_pred = model_prediction(X,weight=weight,bias=bias)
# mse = mean_squared_error(y,y_pred)



