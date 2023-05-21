import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#用于导入 seaborn 库，让图表更丰富
from sklearn.model_selection import cross_val_score,train_test_split
#划分数据与交叉验证模块
from sklearn.preprocessing import StandardScaler
#数据标准化模块
from sklearn.metrics import mean_squared_error,make_scorer
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
train2=pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_regression.csv")
y=train.co_value_log1p
train3=train2[['fint_volume','fint_volume_log','fint_volume-S2','fint_volume-S3','fint_volume-SQ','fint_speed','fint_speed_log','fint_speed-S2','fint_speed-S3','fint_speed-SQ','vol_speed_value','windSpeed_value','windSpeed_value-S2','windSpeed_value-S3','windSpeed_value-SQ','windSpeed_value_log','temp_value','temp_value_S3','temp_value_SQ','temp_value_log','humidity_value','humidity_value_S2','humidity_value_S3','humidity_value_SQ','humidity_value_log']]
X_train, X_test, y_train, y_test = train_test_split(train3, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
# train3.to_csv('H:/北横项目部/风机运行数据/Danalysis/MR_CO_train.csv')
#使用误差度量RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)
#交叉验证（评估方法对象（分类器），数据特征，数据标签，调用方法，交叉验证折数）

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)
#4* ElasticNet
elasticNet = ElasticNetCV(l1_ratio = [0.01,0.05,0.1,0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.001,0.002, 0.003, 0.006,0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                          max_iter = 100000, cv = 10)
elasticNet.fit(X_train, y_train)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio*.1,ratio*.5,ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15,ratio * 1.2],
                          alphas = [0.0001,0.0005,0.001,0.002, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                          max_iter = 100000, cv = 10)
elasticNet.fit(X_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
      " and alpha centered around " + str(alpha))
elasticNet = ElasticNetCV(l1_ratio = ratio,
                          alphas = [alpha*0.5,alpha*0.9,
                                    alpha*0.98,alpha * 1, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3,
                                    alpha * 1.35, alpha * 1.4],
                          max_iter = 100000, cv = 10)
elasticNet.fit(X_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())
print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())
y_train_ela = elasticNet.predict(X_train)
y_test_ela = elasticNet.predict(X_test)

# Plot residuals
plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y=0,xmin=-1.5,xmax=1.5,color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([-2,3],[-1.5,1.5],c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(elasticNet.coef_, index = X_train.columns)
print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")
imp_coefs =coefs.sort_values()
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")
plt.show()
imp_coefs.to_csv("H:/北横项目部/风机运行数据/Danalysis/ElasticNet_coef.csv")