import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
import pandas as pd
import joblib
#使用joblib库保存随机森林模型
# data missing value treatment
from sklearn.impute import SimpleImputer
#data processing
#data transformation-categorical
from sklearn.preprocessing import OneHotEncoder
#data transformation-numerical
from sklearn.preprocessing import StandardScaler
#train and test splite
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve
#build model
from sklearn.ensemble import RandomForestRegressor
#Evaluate Model
from sklearn.metrics import mean_squared_error,r2_score #use squared False for RMSE
#混淆矩阵
from sklearn.metrics import confusion_matrix
import sklearn

#二、输入数据
train = pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_EDN4(y1_1hours).csv")
y=train.co_value_log
train2=pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_EDN4(x1_1hours).csv")
train3=train2[['windSpeed_value','fan_off','fint_volume-SQ','fint_speed-S2','congestion','workday','co_value_1hours']]
#拆分训练集和验证集
# train3.drop(columns=['fint_speed'])
x_train,x_test,y_train,y_test=train_test_split(train3,y,test_size = 0.3, random_state = 0)
print("X_train : " + str(x_train.shape))
print("X_test : " + str(x_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
#10/通过GridSearchCV进行参数调优
param_grid = [{'n_estimators': [3, 10, 30,60,80,100], 'max_features': [2,3,4,5],'bootstrap': [True,False], 'random_state':[0],}]
#11/创建模型
RF_get=RandomForestRegressor()
rf=GridSearchCV(RF_get,param_grid,cv=10,scoring='neg_mean_squared_error')
#Fit
rf.fit(x_train,y_train)
RMSE_results = pd.DataFrame(rf.cv_results_)
RMSE_results['ID']=range(1,len(RMSE_results)+1)
RMSE_results['RMSE'] = np.sqrt(-RMSE_results[['mean_test_score']])

# 绘制RMSE曲线
fig,ax=plt.subplots(figsize=(10,6))
ax.plot(RMSE_results['ID'], RMSE_results['RMSE'], "bo-")
ax.set_xlabel("Parameter Combination")
ax.set_ylabel("RMSE")
ax.set_title("RMSE for Each Parameter Combination")
print(plt.show())
#Predict
#train
train_predict=rf.predict(x_train)
test_predict=rf.predict(x_test)
#生成学习曲线
r2_score(y_train,train_predict)
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestRegressor(bootstrap=True, max_features=2, n_estimators=100, random_state= 0), x_train, y_train, train_sizes=np.linspace(0.001, 0.3, 100),
    scoring="r2", cv=10, shuffle=True, random_state=42,n_jobs=4)
#绘制学习曲线
fig1,ax=plt.subplots(1,1,figsize=(6,6))
# ax.set_ylim((0.7,1.1)) # 设置子图的纵坐标的范围为（0.7~1.1）
ax.set_xlabel("training examples") # 设置子图的x轴名称
ax.set_ylabel("score")
ax.grid() # 画出网图
ax.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color='r',label='train score')
# 画训练集数据分数，横坐标为用作训练的样本数，纵坐标为不同折下的训练分数的均值
ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='g',label='test score')
ax.legend(loc='best') # 设置图例
plt.show()
print(rf.best_params_)
print(train_predict)
# test
print(test_predict)
plt.scatter(train_predict, train_predict - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(test_predict, test_predict - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("RF regression")
plt.xlabel("predict values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y = 0, xmin = -2, xmax = 3, color = "red")
print(plt.show())
plt.scatter(train_predict, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(test_predict, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([-2,3],[-2,3],c = "red")
print(plt.show())
#11/评估
#Train RMSE
print(np.sqrt(mean_squared_error(train_predict,y_train)))
#Test RMSE
print(np.sqrt(mean_squared_error(test_predict,y_test)))
#Train R**2
print(r2_score(y_train,train_predict))
#Test R**2
print(r2_score(y_test,test_predict))
#12/预测并提交
#predict
# predict=pd.DataFrame({"CO_value": test_predict})
# predict.to_csv("H:/kaggle-study/RF/RF_CO.csv",index=False)
# workday=train2[train2['workday']==1]
# new_data=workday.groupby('time')[['fint_volume-SQ','fint_speed-S2']].mean()
# data1=train['time']
# data=new_data.reset_index(drop=True)
# data['fan_off']=0
# data['congestion']=0
# data['workday']=1
# data=data.reindex(columns=['workday','fan_off','fint_volume-SQ','fint_speed-S2','congestion'])
# predict_co=rf.predict(data)
# predict_co1=pd.DataFrame(np.exp(predict_co),columns=['co'])
# predict=pd.concat([predict_co1,data],axis=1)
# co=predict.set_index(data1)
# co.to_csv('H:/北横项目部/风机运行数据/Danalysis/MR_CO_predict1.csv')
#13保存训练模型
joblib.dump(rf, 'H:/北横项目部/风机运行数据/Danalysis/rf.pkl')



