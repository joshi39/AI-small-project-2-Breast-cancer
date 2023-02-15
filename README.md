# AI-small-project-2-Breast-cancer
This code performs a classification task using Support Vector Machines (SVMs) to classify breast cancer as either malignant or benign. Here is a brief summary of what the code does:
Loads the breast cancer dataset using the scikit-learn library.
Prints the names of the features and the label types of the cancer (malignant or benign).
Splits the dataset into training and test sets using train_test_split() from scikit-learn.
Defines hyperparameters to be tuned using grid search cross-validation.
Creates a SVM classifier using scikit-learn's SVC class.
Performs grid search cross-validation using GridSearchCV to find the best hyperparameters for the SVM classifier.
Retrains the SVM classifier on the entire training set using the best hyperparameters.
Predicts the class labels for the test set using the trained SVM classifier.
Calculates evaluation metrics (accuracy, precision, recall, and specificity) for the predicted labels using scikit-learn's metrics module.
Prints the best hyperparameters found by grid search, as well as the evaluation metrics.
Overall, this code provides an example of how to use scikit-learn to perform a classification task using SVMs and how to tune hyperparameters using grid search cross-validation.
