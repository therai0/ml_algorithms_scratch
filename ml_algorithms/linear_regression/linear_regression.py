
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error



class LR:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test
    
    @staticmethod
    def model_evaluation(y_test,y_pred)->dict:
        try:
            return {
                "r2_score":r2_score(y_test,y_pred),
                "mean_absolute_error":mean_absolute_error(y_test,y_pred),
                "mean_squared_error":mean_absolute_error(y_test,y_pred)
            }
        except Exception as e:
            print(e) 
    

    def linear_regression(self)->dict:
        try:
            model = LinearRegression()
            model.fit(self.X_train,self.y_train)

            y_pred = model.predict(self.X_test)

            model_evl = self.model_evaluation(y_test,y_pred)
            return model_evl 
        except Exception as e:
            print(e)




if __name__ == "__main__":
    X,y = make_regression(n_samples=1000,n_features=3,n_informative=1,n_targets=1,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
   
    lr = LR(X_train,y_train,X_test,y_test)

    result = lr.linear_regression()

    print(result)

