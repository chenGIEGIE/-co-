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
#njobs=4
# Get data
train = pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_ED.csv")
print("train : " + str(train.shape))
# Check for duplicates
idsUnique = len(set(train.COtime))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
#查找异常值，参考https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
plt.scatter(train.fint_speed,train.fint_volume,c="blue",marker=".")
plt.title("looking for outliers")
plt.xlabel("fint_speed")
plt.ylabel("fint_volume")
plt.show()
#根据车速车流分析，车速在小于40的情况下，车流量存在断档，该部分数据出现在夜间22:55至5:00，故推测为作业车辆数据干扰