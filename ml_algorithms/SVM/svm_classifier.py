from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class SupportVectorClassifier:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test 

    @staticmethod
    def model_evaluation(y_test,y_pred):
        try:
            report = classification_report(y_test,y_pred)
            return report 
        except Exception as e:
            print(e)


    def support_vector_classifier(self):
        try:
            svc = SVC()
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
    df = load_iris()
    X = df.data
    y = df.target 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    sv_c = SupportVectorClassifier(X_train,X_test,y_train,y_test)

    sv_c.support_vector_classifier()