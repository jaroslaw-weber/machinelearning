import sklearn as sk
import autosklearn.classification
import random
import numpy as np

#### testing popular AutoML library (autosklearn) with simple classifier function

#we are training classifier to guess this classification function.
def function_to_guess(a, b):
    if(a<50 and b<50): return 1
    if(a>150 and b>150): return 3
    return  2

#define helper functions to create data
def create_x():
    return np.random.randint(0,200,size=(100, 2))
def create_y(_x):
    return [function_to_guess(a,b) for a,b in _x]

print("creating data")
x= create_x()
print(x[:3])
y= create_y(x)
print(y[:3])
xtest=create_x()
ytest=create_y(xtest)

#first lets test it with Random Forest
rf_model=sk.ensemble.RandomForestClassifier()
print("training model:")
rf_model.fit(x,y)
print("score:")
print(rf_model.score(xtest, ytest))

#create model
model=autosklearn.classification.AutoSklearnClassifier()
print("training model")
model.fit(x,y)
print("score")
print(model.score(xtest, ytest))

# As we can see, its much faster to use defined model rather than AutoML library.
# Furthermore, I encountered some issues with AutoML library (lot of errors and warnings)
# Conclusion: it is probably more faster and efficient to test and tweak standard sklearn models.
# In this case Random Forest reach 95% accuracy, which is enough for most cases.