import numpy as np 
def mean_squared_error(y,y_pred):
    try:
        l = len(y)
        diff = []

        # if single data point is pass 
        if l == 1:
            return np.power((y - y_pred),2)
        
        for i in range(len(y)):
            diff.append(np.power((y[i]-y_pred[i]),2))

        return sum(diff)/l
    except Exception as e:
        print(e)