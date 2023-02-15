from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import svm
from sklearn import metrics

# Load the breast cancer dataset
bcw = datasets.load_breast_cancer()

# print the names of the 13 features
print("Features: ", bcw.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", bcw.target_names)

# Spliting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(bcw.data, bcw.target, test_size=0.3, random_state=109)

# Defining the hyperparameters to tune
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma':[0.1, 1, 10]}

# creating SVM classifier
clf = svm.SVC()

# using gridsearch to hypertune
grid_search = GridSearchCV(clf, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Train the SVM classifier on the entire training set using the best hyperparameters
clf = svm.SVC(**grid_search.best_params_)
clf.fit(X_train, y_train)

# Use the trained classifier to predict the class labels of the test set
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
specificity = metrics.recall_score(y_test, y_pred, pos_label=0)

# printing final performance of the classifier
print("Best hyperparameters:", grid_search.best_params_)
print("Test set accuracy:", accuracy)
print("Test set precision:", precision)
print("Test set recall:", recall)
print("Test set specificity:", specificity)
