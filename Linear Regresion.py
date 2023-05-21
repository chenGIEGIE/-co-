import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#用于导入 seaborn 库，让图表更丰富
from sklearn.model_selection import cross_val_score,train_test_split
#划分数据与交叉验证模块
from sklearn.preprocessing import StandardScaler
#数据标准化模块
from sklearn.metrics import mean_squared_error,make_scorer,r2_score
#载入均方误差与评分函数模块
from scipy.stats import skew
#计算数据集的样本偏度模块
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
#载入线型模型模块，包括线型回归、岭回归、Lasso回归、ElasticNetCV
from IPython.display import display
#display该函数运行适当的dunder（魔法方法，如构造函数__init__）以获取适当的数据以显示
from warnings import simplefilter
#载入报警忽略模块
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 100)
#划分训练集和测试集
train = pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_regression_y.csv")
train3=pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_regression.csv")
y=train.co_value_log1p
X_train, X_test, y_train, y_test = train_test_split(train3, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
train3.to_csv('H:/北横项目部/风机运行数据/Danalysis/MR_CO_train.csv')
#使用误差度量RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)
#交叉验证（评估方法对象（分类器），数据特征，数据标签，调用方法，交叉验证折数）

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)
# 线型回归
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
# print("方程为 y={w1}x1+{w2}x2+{w3}x3+{w4}x4+{w5}x5+{w6}x6+{w7}x7+{w8}x8+{w9}x9+{w10}x10+{w11}x11+{w12}x12+{w13}x13",)
#sklearn里的封装好的各种算法使用前都要fit，有监督学习的算法fit(x,y)传两个参数。无监督学习的算法是fit(x)，即传一个参数，比如降维、特征提取、标准化。
#查看训练和验证集的预测
print("RMSE on Training set :", rmse_cv_train(lr).mean())
print("RMSE on Test set :", rmse_cv_test(lr).mean())
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
#Train R**2
print(r2_score(y_train,y_train_pred))
#Test R**2
print(r2_score(y_test,y_test_pred))
# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = -2, xmax = 4, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([-1.5,1.5],[-2,3],c = "red")
plt.show()