# Sample of trainning model
# ZYF
# 1）导入数据集

import numpy as np
import pandas as pd

data = pd.read_csv("./data.csv")


train = data[:-50]
test = data[-50:]

train_x = train.values[:,:-1]
train_y = train.values[:,-1:]

test_x = test.values[:,:-1]
test_y = test.values[:,-1:]

print("Shape of train x : {}".format(train_x.shape))
print("Shape of train y : {}".format(train_y.shape))
print("Shape of test x : {}".format(test_x.shape))
print("Shape of test y : {}".format(test_y.shape))




# 2）调参训练模型

from Tool_20210514 import model

m = model(path_model_csv="./results/",path_model_pic="./pic/",path_best_model="./models/")
model_name = "Lr"
name = "test"

m.load_data(train_x,test_x,train_y,test_y,train_x.shape[1])
m.run(model_name=model_name,name=name)
