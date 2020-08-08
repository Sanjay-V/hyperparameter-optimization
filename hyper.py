from sklearn import svm, datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
data = pd.DataFrame(iris.target)
print(data)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
print(df[47:150])
#Approach 1: Use train_test_split and manually tune parameters by trial and error
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
print(model.score(X_test, y_test))
#Approach 2: Use GridSearchCV
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
print(clf.cv_results_)
df = pd.DataFrame(clf.cv_results_)
print(df)
print(df[['param_C','param_kernel','mean_test_score']])
print(clf.best_params_)
print(clf.best_score_)
#print(dir(clf))
