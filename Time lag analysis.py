import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
train=pd.read_csv("H:/北横项目部/风机运行数据/Danalysis/MR_CO_EDN.csv")
#划分图表
train["co_value_new"]=train["co_value"].shift(12)
train["fint_speed_new"]=train["fint_speed"].shift(12)
train["fint_volume_new"]=train["fint_volume"].shift(12)
train["temp_value_new"]= train["temp_value"].shift(12)
train["humidity_value_new"]=train["humidity_value"].shift(12)
train["windSpeed_value_new"]=train["windSpeed_value"].shift(12)
train["fan_off_new"]=train["fan_off"].shift(12)
train["TIME_new"]=train["TIME"].shift(12)
train=train.reindex(columns=["co_value","co_value_new","fint_speed","fint_speed_new","fint_volume","fint_volume_new","temp_value","temp_value_new","humidity_value","humidity_value_new","windSpeed_value","windSpeed_value_new","fan_off","fan_off_new","TIME","TIME_new"])
plt.subplot(4,4,1)
#绘制数据和线性回归拟合
re=sns.regplot(data=train,x="co_value_new",y="co_value",ci=95, scatter_kws={"color":"green","s":2,"alpha":0.3})
re.set_title('co plot of new co')
#co和他的滞后1阶相关性
plt.subplot(4,4,5)
re1=sns.regplot(data=train,x="co_value",y="co_value",ci=95, scatter_kws={"color":"green","s":2,"alpha":0.3})
re1.set_title('co plot of co')
#co和他本身相关性
plt.subplot(4,4,2)
re2=sns.regplot(data=train,x="co_value",y="fint_volume_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re2.set_title("xco-yvolume_new")
#co和车流量滞后一阶相关性
plt.subplot(4,4,6)
re3=sns.regplot(data=train,x="co_value",y="fint_volume",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re3.set_title("xco-yvolume")
#co和车流量相关性
plt.subplot(4,4,3)
re4=sns.regplot(data=train,x="co_value",y="fint_speed_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re4.set_title("xco-yspeed_new")
#co和车速滞后1阶相关性
plt.subplot(4,4,7)
re5=sns.regplot(data=train,x="co_value",y="fint_speed",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re5.set_title("xco-yspeed")
#co和车速相关性
plt.subplot(4,4,4)
re6=sns.regplot(data=train,x="co_value",y="temp_value_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re6.set_title("xco-ytemp_new")
#co和temp_value一阶滞后
plt.subplot(4,4,8)
re7=sns.regplot(data=train,x="co_value",y="temp_value",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re7.set_title("xco-ytemp")
#co和temp_value相关
plt.subplot(4,4,9)
re8=sns.regplot(data=train,x="co_value",y="humidity_value_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re8.set_title("xco-yhumidity_new")
#co和humidity一阶滞后
plt.subplot(4,4,13)
re9=sns.regplot(data=train,x="co_value",y="humidity_value",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re9.set_title("xco-yhumidity")
#co和humidity相关
plt.subplot(4,4,10)
re8=sns.regplot(data=train,x="co_value",y="windSpeed_value_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re8.set_title("xco-ywindSpeed_new")
#co和windSpeed一阶滞后
plt.subplot(4,4,14)
re9=sns.regplot(data=train,x="co_value",y="windSpeed_value",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re9.set_title("xco-ywindSpeed")
#co和windSpeed相关
plt.subplot(4,4,11)
re8=sns.regplot(data=train,x="co_value",y="fan_off_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re8.set_title("xco-yfan_off_new")
#co和fan_off一阶滞后
plt.subplot(4,4,15)
re9=sns.regplot(data=train,x="co_value",y="fan_off",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re9.set_title("xco-yfan_off")
#co和fan_off相关
plt.subplot(4,4,12)
re8=sns.regplot(data=train,x="co_value",y="TIME_new",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re8.set_title("xco-yTIME_new")
#co和time一阶滞后
plt.subplot(4,4,16)
re9=sns.regplot(data=train,x="co_value",y="TIME",ci=95,scatter_kws={"color":"green","s":2,"alpha":0.3})
re9.set_title("xco-yTIME")
plt.show()
#co和time相关
#根据分析结果，车流车速在进行时间滞后之后与CO浓度表现出更好的线性相关，时间变量表现出周期性。