from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split


class SupportVectorRegression:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test 

    @staticmethod
    def model_evaluation(y_test,y_pred):
        try:
            r2 = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
            return [r2,mse] 
        except Exception as e:
            print(e)


    def support_vector_regression(self):
        try:
            svc = SVR()
            scalar = StandardScaler()
            X_scaled_train = scalar.fit_transform(self.X_train)
            X_scaled_test = scalar.transform(self.X_test)

            svc.fit(X_scaled_train,self.y_train)
            y_pred = svc.predict(X_scaled_test)

            report  = self.model_evaluation(self.y_test,y_pred)

            print(report)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    X,y = make_regression(n_samples=1000,n_features=5,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    sv_r = SupportVectorRegression(X_train,X_test,y_train,y_test)

    sv_r.support_vector_regression()