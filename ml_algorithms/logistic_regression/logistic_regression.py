from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class LoR:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train= X_train
        self.X_test =X_test 
        self.y_train = y_train 
        self.y_test = y_test 
    
    @staticmethod
    def model_evaluation(y_test,y_pred):
        try:
            return classification_report(y_test,y_pred)
        except Exception as e:
            print(e)    


    def logistic_regression(self):
        try:
            lr = LogisticRegression()
            lr.fit(self.X_train,self.y_train)

            y_pred = lr.predict(self.X_test)

            report = self.model_evaluation(self.y_test,y_pred)
            return report 
        except Exception as e:
            print(e)

if __name__ == "__main__":
    X,y = make_classification(n_samples=1000,n_features=4,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    lr = LoR(X_train,X_test,y_train,y_test)

    report = lr.logistic_regression()

    print(report)