# --- Simple implementation of AutoML for classification problem -----
# When we want to classify data we must decide on classifier type.
# But we should have to tweak values for every single problem.
# ML is a concept of detecting best ML tool to solve a problem.
# This code is implementation of simple AutoML tool.

#importing iris dataset
from sklearn.datasets import load_iris

#cross validation tool
from sklearn.model_selection import train_test_split

#importing classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.svm import SVC

#### preprocessing data

#load data
iris= load_iris()
#split data to train and test set
x, x_test, y, y_test =  train_test_split(iris.data, iris.target, test_size=0.35, random_state=0)

#show how big is train data. if not enough you should adjust test_size
print("data size:")
print(len(x))

#function for testing single models. it print and returns a score
def testmodel(label, model):
    #train model
    model.fit(x,y)
    #estimate model with test data
    score=model.score(x_test, y_test)
    #print score
    print("model training results. model: {0}, score: {1}".format(label, score))
    #return score
    return score

#define what models you want to test
models_to_test=\
    [
     ("SDG", SGDClassifier()),
     ("KNN", KNeighborsClassifier()),
     ("Decision Tree", DecisionTreeClassifier()),
     ("SVC (SVM)", SVC())
    ]

#cache best model info here
best_model=("error: models not defined", 0)

#test all defined models
for label, model_to_test in models_to_test:
    score=testmodel(label, model_to_test)
    if score>best_model[1]:
        best_model=(label, score)

#lets print best model
best_model_name, best_score=best_model
print ("best model: {0}, with score: {1}".format(best_model_name, best_score))

## comments:
# looks like both KNN, DecisionTree and SVC works nicely with small amounts of data.
# Vanilla SDG was recommended as a good classifier on Sklearn cheetsheet,
# but it looks like there are better alternatives for our dataset.
# To improve this AutoML classifier selection you could tweak hyperparameters
# and add more types of classifiers and cross validations.


