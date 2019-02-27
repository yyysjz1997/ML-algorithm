import numpy as np
from sklearn import datasets
from sklearn.cross_validation import  train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
#  knn处理自带数据库(iris花的数据)
iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target

#print(iris_X[:3,:])
#print(iris_y)

X_train,X_test,y_train,y_test = train_test_split(
    iris_X,iris_y,test_size=0.3)

#print(y_train)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

print(knn.predict(X_test))
print(y_test)
'''

from sklearn.linear_model import LinearRegression

'''
#  线性回归
loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])

print(model.coef_) # y=kx+b   coef_ = k,intercept_ = b
print(model.intercept_)
print(model.score(data_X,data_y)) #R^2的偏差，即准确度
'''

import matplotlib.pyplot as plt
'''
#  自己建立数据库
X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(X,y)  #点输出结果
plt.show()
'''



from sklearn import preprocessing
'''
#  标准化
a=np.array([[10,2.7,3.6],[-100,5,-2],[120,20,40]],dtype=np.float64)
print(a)
print(preprocessing.scale(a))
'''

from sklearn.datasets.samples_generator import  make_classification
from sklearn.svm import SVC
'''
#  标准化后，分类
X,y=make_classification(n_samples=300,n_features=2,n_redundant=0,
                        n_informative=2,random_state=22,n_clusters_per_class=1,scale=100)
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.show()
X=preprocessing.scale(X)
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3)
clf=SVC()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
'''

from sklearn.datasets import load_iris
'''
#   交叉验证
iris=load_iris()
X=iris.data
y=iris.target
#X_train,X_test,y_train,y_test = train_test_split(
 #   X,y,random_state=4)
#knn=KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train,y_train)
#print(knn.score(X_test,y_test))

from sklearn.cross_validation import cross_val_score
k_range=range(1,31)
k_scores=[]
for k in k_range:
    #n_neighbors为查询几个最邻近的数目
    knn=KNeighborsClassifier(n_neighbors=k)
    #  自动分为10组cv=10即是10组
    loss=-cross_val_score(knn,X,y,cv=10,scoring='mean_squared_error')
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
    #print(scores)
    #print(scores.mean())#mean为取上述五个的平均值

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
'''

from sklearn.learning_curve import  learning_curve
from sklearn.datasets import load_digits
#实时记录学习的过程，train_sizes中有几个值即为查看几次
'''
digits=load_digits()
X=digits.data
y=digits.target

train_size,train_loss,test_loss=learning_curve(
    SVC(gamma=0.001),X,y,cv=10,scoring='mean_squared_error',
    train_sizes=[0.1,0.25,0.5,0.75,1])
train_loss_mean=-np.mean(test_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(train_size,train_loss_mean,'o-',color="r",
         label="Training")
plt.plot(train_size,test_loss_mean,'o-',color="g",
         label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
'''

from sklearn.learning_curve import  validation_curve
'''
#对于过拟合的处理,选取哪一个gamma是最好的
digits=load_digits()
X=digits.data
y=digits.target
param_range=np.logspace(-61,-2.3,5)

train_loss,test_loss=validation_curve(
    SVC(),X,y,param_name='gamma',param_range=param_range,
    cv=10,scoring='mean_squared_error')
train_loss_mean=-np.mean(test_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(param_range,train_loss_mean,'o-',color="r",
         label="Training")
plt.plot(param_range,test_loss_mean,'o-',color="g",
         label="Cross-validation")
plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
'''


from sklearn import svm
from sklearn import datasets
'''
#  保存model
clf=svm.SVC()
iris=datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)

#方法一：pickle
#import pickle
#with open('save/clf.pickle','wb')as f:
#    pickle.dump(clf,f)
#with open('save/clf.pickle','rb')as f:
#    clf2=pickle.load(f)
#    print(clf2.predict(X[0:1]))

#方法二：joblib
from sklearn.externals import  joblib
#Save
joblib.dump(clf,'clf.plk')
#restore
clf3=joblib.load('clf.plk')
print(clf3.predict(X[0:1]))
'''