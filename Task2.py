import numpy as np
from scipy.io import loadmat
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LogisticRegression

#~1:30 runtime to load data
#X_train = np.loadtxt("Task2Data\X_train", dtype=np.dtype(int), delimiter=",")
#X_test = np.loadtxt("Task2Data\X_test", dtype=np.dtype(int), delimiter=",")
#y_train = np.loadtxt("Task2Data\y_train", dtype=np.dtype(int), delimiter=",")
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)

data = loadmat("data.mat")
X_train = data["X_train"]
X_submission_test = data["X_test"]
y_train = data["y_train"]
y_train = y_train.T

print(X_train.shape)
print(y_train.shape)
print(X_submission_test.shape)

#train test ssplit
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1)

#normalize data
X_train = np.divide(X_train, np.tile(np.reshape(np.linalg.norm(X_train, axis = 1), (X_train.shape[0], 1)), (1, 90)))
X_submission_test = np.divide(X_submission_test, np.tile(np.reshape(np.linalg.norm(X_submission_test, axis = 1), (X_submission_test.shape[0], 1)), (1, 90)))

num_trees = 50
model = RandomForestClassifier(n_estimators = num_trees, criterion = "gini", max_samples = None, max_features = 1.0, max_depth = 9, min_samples_leaf = 100)

#resample data
ros = RandomOverSampler(random_state = 0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

#logistic regression
X_test_resampled, y_test_resampled = ros.fit_resample(X_test, y_test)
y_train = y_train.reshape((y_train.shape[0]))
y_test = y_test.reshape((y_test.shape[0]))
y_resampled = y_resampled.reshape((y_resampled.shape[0]))
y_test_resampled = y_test_resampled.reshape((y_test_resampled.shape[0]))
logistic = LogisticRegression(penalty = "l2", class_weight="balanced", max_iter = 1000).fit(X_train, y_train)


log_train_pred = logistic.predict(X_resampled)
log_test_pred = logistic.predict(X_test_resampled)

log_train_error_mean = mean_absolute_error(y_resampled, log_train_pred)
log_train_error_median = median_absolute_error(y_resampled, log_train_pred)
log_test_error_mean = mean_absolute_error(y_test_resampled, log_test_pred)
log_test_error_median = median_absolute_error(y_test_resampled, log_test_pred)

print("LOGISTIC oversampled:")
print("train mean {}".format(log_train_error_mean))
print("train medi {}".format(log_train_error_median))
print("")
print("test mean {}".format(log_test_error_mean))
print("test medi {}".format(log_test_error_median))

##"regular" sampled error
#model.fit(X_resampled, y_resampled)
#train_pred = model.predict(X_train)
#test_pred = model.predict(X_test)

#train_error_mean = mean_absolute_error(y_train, train_pred)
#train_error_median = median_absolute_error(y_train, train_pred)
#test_error_mean = mean_absolute_error(y_test, test_pred)
#test_error_median = median_absolute_error(y_test, test_pred)

#print("\"regular\" sampled data (no oversampling):")
#print("train mean {}".format(train_error_mean))
#print("train medi {}".format(train_error_median))
#print("")
#print("test mean {}".format(test_error_mean))
#print("test medi {}".format(test_error_median))
#print("\n")

##oversampled error
#train_oversampled_pred = model.predict(X_resampled)
#X_test_resampled, y_test_resampled = ros.fit_resample(X_test, y_test)
#test_oversampled_pred = model.predict(X_test_resampled)

#train_oversampled_error_mean = mean_absolute_error(y_resampled, train_oversampled_pred)
#train_oversampled_error_median = median_absolute_error(y_resampled, train_oversampled_pred)
#test_oversampled_error_mean = mean_absolute_error(y_test_resampled, test_oversampled_pred)
#test_oversampled_error_median = median_absolute_error(y_test_resampled, test_oversampled_pred)

#print("oversampled data:")
#print("train mean {}".format(train_oversampled_error_mean))
#print("train medi {}".format(train_oversampled_error_median))
#print("")
#print("test mean {}".format(test_oversampled_error_mean))
#print("test medi {}".format(test_oversampled_error_median))

##save predictions to file
#submission_pred = model.predict(X_submission_test)
#print("X_submission_test.shape[0]: {0}. submission_predictions.shape: {1}".format(X_submission_test.shape[0], submission_pred.shape))
#np.savetxt("C:\\Users\\Steven\\Desktop\\CS 471\\Projects\\Final Project\\final-project-stentann\\task_2\\y_pred.gz", submission_pred.tolist(), delimiter=",")
##predicted error as oversampled median
#print("predicted_loss type: {0}".format(type(test_oversampled_error_median)))
#error_prediction = (np.ones((1)) * test_oversampled_error_median).tolist()
#print("error type: {}".format(type(error_prediction[0])))
#np.savetxt("C:\\Users\\Steven\\Desktop\\CS 471\\Projects\\Final Project\\final-project-stentann\\task_2\\err_pred.txt", error_prediction)

print("end")