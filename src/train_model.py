import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import recall_score,accuracy_score,f1_score,precision_score

from load_data import DataLoader

def main():
    # Train the random forest model
    X_train, X_test, y_train, y_test = DataLoader().load()
    clf = RandomForestRegressor(n_estimators = 75,max_depth=4)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    y_pred=(y_pred>0.5)

    y_pred=clf.predict(X_test)
    y_pred=(y_pred>0.5)
    y_test=list(y_test)
    y_pred=list(y_pred)

    with open('model.pkl', 'wb') as handle:
        pickle.dump(clf, handle)

    print("Accuracy for Random Forest Regressor on Test Data= "+str(accuracy_score(y_test,y_pred)*100))
    print("Test F1: "+str(f1_score(y_test, y_pred)*100))
    print("Test recall: "+str(recall_score(y_test, y_pred)*100))
    print("Test precision: "+str(precision_score(y_test, y_pred)*100))

if __name__=='__main__':
    main()
