"""
@project: Tool Kit for Improving Experimental Efficiency
@autor：Yufan Zheng
@file：Tool.py
@time：2021-09-20
@vision：1.12
@email: zhjpre@gmail.com
"""

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,mean_squared_log_error
import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from itertools import combinations
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import csv
from sklearn.svm import SVR,LinearSVR
from numpy import random
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,SGDRegressor,BayesianRidge,LogisticRegression,PassiveAggressiveRegressor,ElasticNet,Ridge,Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
import joblib
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score,silhouette_score



# 寻找文件夹下所有文件的名字,suffix为指定后缀进行查找，如 suffix=".csv"
def file_name(file_dir,suffix=None): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if(suffix==None):
                L.append(os.path.join(root, file).replace("\\","/"))
            else:
                if (os.path.splitext(file)[1] == suffix):
                    L.append(os.path.join(root, file).replace("\\","/"))
    return L
# ---------------------------------------分割线------------------------------------------------

# 创建文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")
# ---------------------------------------分割线------------------------------------------------

'''
将正常数据集转换为时间序列数据
Input:
    data：原始数据（按时间排序）；使用num天的数据去预测later天后的值，type:DataFrame；
    ignore：不对其进行时间序列操作的列名，如一些固化不变（不随时间变化）的特征（经纬度），type:List；
    y：预测目标的列名，type：string；
    drop_c：需要在操作前删除的列，type：List；
    is_sum：预测的时间单位的大小，默认是1，如代表预测某一天。若为7，则代表预测连续7天的累计值。
Output:
    seq：创建好重构后的时间序列格式数据。
'''
def create_time_seq(data,num=2,later=1,ignore=None,y=None,drop_c=None,is_sum=1):
    if(drop_c!=None):
        data = data.drop(columns=drop_c)
    later = later-1
    if(isinstance(data,pd.DataFrame)): # 先判断类型是否是pd.DataFrame，不是的话直接退出
        if(ignore!=None and y!=None):
            print("1")
            # ingure 和 y 都有指定
            # 取出y值
            if(is_sum==1):
                seq = pd.DataFrame(data[y].values[num+later+is_sum:].tolist(),columns=["Y"])
            else:
                sy = []
                for n in range(data.shape[0]-later-is_sum-num):
                    sy.append(np.sum(data[y].iloc[n+later+num+1:n+is_sum+later+num+1].values))
                seq = pd.DataFrame(sy,columns=["Y"])
            for c in data.columns: # 遍历所有的列名
                if c not in ignore: # 判断列名是否在ignore里
                    cl = data[c].values # 取出相应列名的列
                    cl_time = np.array([cl[n:n+num] for n in range(cl.shape[0]-num-later-is_sum)]) # 按每次取num行出来，直到倒数第num+later+1行结束
                    cl_name = [c+"_"+str(i+1) for i in range(num)] # 取列名
                    data2 = pd.DataFrame(cl_time.reshape(-1,num),columns=cl_name)
                    seq = pd.concat([data2,seq],axis=1) # 跟之前存在的数据进行拼接
                else:
                    seq[c] = data[c].values[num+later+is_sum:]
        elif(ignore!=None and y==None):
            print("2")
            # ingure有指定，但是y没指定，则取最后一列
            # 取出y值
            y = -1
            if(is_sum==1):
                seq = pd.DataFrame(data.iloc[num+later+is_sum:,-1].values.tolist(),columns=["Y"])
            else:
                sy = []
                for n in range(data.shape[0]-later-is_sum-num):
                    sy.append(np.sum(data[y].iloc[n+later+num+1:n+is_sum+later+num+1].values))
                seq = pd.DataFrame(sy,columns=["Y"])
            for c in data.columns:
                if c not in ignore:
                    cl = data[c].values
                    cl_time = np.array([cl[n:n+num] for n in range(cl.shape[0]-num-later-is_sum)])
                    cl_name = [c+"_"+str(i+1) for i in range(num)]
                    data2 = pd.DataFrame(cl_time.reshape(-1,num),columns=cl_name)
                    seq = pd.concat([data2,seq],axis=1)
                else:
                    seq[c] = data[c].values[num+later+is_sum:]
        elif(ignore==None and y!=None):
            print("3")
            # y有指定，但是ingure没指定，则取除去y列剩下的所有
            # 取出y值
            if(is_sum==1):
                seq = pd.DataFrame(data[y].values[num+later+is_sum:].tolist(),columns=["Y"])
            else:
                sy = []
                for n in range(data.shape[0]-later-is_sum-num):
                    sy.append(np.sum(data[y].iloc[n+later+num+1:n+is_sum+later+num+1].values))
                seq = pd.DataFrame(sy,columns=["Y"])
            for c in data.columns:
                cl = data[c].values
                cl_time = np.array([cl[n:n+num] for n in range(cl.shape[0]-num-later-is_sum)])
                cl_name = [c+"_"+str(i+1) for i in range(num)]
                data2 = pd.DataFrame(cl_time.reshape(-1,num),columns=cl_name)
                print(data2.shape[0])
                seq = pd.concat([data2,seq],axis=1)
        elif(ignore==None and y==None):
            print("4")
            # y , ingure都没指定，则取最后一列为y,其他所有特征都做成时序数据
            # 取出y值
            if(is_sum==1):
                seq = pd.DataFrame(data.iloc[num+later+is_sum:,-1].values.tolist(),columns=["Y"])
            else:
                sy = []
                for n in range(data.shape[0]-later-is_sum-num):
                    sy.append(np.sum(data[y].iloc[n+later+num+1:n+is_sum+later+num+1].values))
                seq = pd.DataFrame(sy,columns=["Y"])
            for c in data.columns:
                cl = data[c].values
                cl_time = np.array([cl[n:n+num] for n in range(cl.shape[0]-num-later-is_sum)])
                cl_name = [c+"_"+str(i+1) for i in range(num)]
                data2 = pd.DataFrame(cl_time.reshape(-1,num),columns=cl_name)
                seq = pd.concat([data2,seq],axis=1)   
        print("shape:{}".format(seq.shape))
    else:
        print("Error: type is not pd.DataFrame.")
        return
    return seq
# ---------------------------------------分割线------------------------------------------------



# 7z 压缩数据
import subprocess as sp
import re
import os

_SZIP_PATH = 'C:/Program Files/7-Zip/7z.exe'
_SZ_LIST_PTTRN = re.compile(r'testing +[\w\\./ &!#@]+', re.I)
_SZ_LIST_STRIP = re.compile(r'Testing +', re.I)

comp7z = '7z'
compZip = 'zip'

class SevenZipError(Exception):
    def __init__(self, msg='7z: unknown error'):
        self.value = msg

    def __str__(self):
        return repr(self.value)

class FatalError(SevenZipError):
    def __init__(self, msg='7z: fatal error'):
        self.value = msg

class CommandLineError(SevenZipError):
    def __init__(self, msg='7z: command line error'):
        self.value = msg

class MemError(SevenZipError):
    def __init__(self, msg='7z: not enough memory to perform this operation'):
        self.value = msg

class UserInterrupt(SevenZipError):
    def __init__(self, msg='7z: process interrupted by user'):
        self.value = msg

def unpack(path, targetdir=os.getcwd(), fullpaths=True):
    if not os.path.exists(path):
        raise IOError('Could not find file/directory at path')
    args = [_SZIP_PATH, 'x', path, '-y', '-o' + targetdir]
    if not fullpaths:
        args[1] = 'e'
    sub = sp.Popen(args, stdout=sp.PIPE, stdin=sp.PIPE)
    res = sub.wait()
    if res == 0:
        return True
    elif res == 1:
        return False
    elif res == 2:
        raise FatalError
    elif res == 7:
        raise CommandLineError
    elif res == 8:
        raise MemError
    elif res == 255:
        raise UserInterrupt
    else:
        raise SevenZipError

def unpack_no_full_paths(path, targetdir=os.getcwd()):
    return unpack(path, targetdir, False)

def pack(path, archivepath, compression_level=5, compression_type='7z', password=None):
    if not os.path.exists(path):
        raise IOError('Could not find file/directory at path')
    args = [_SZIP_PATH, 'a', archivepath, path, '-t' + compression_type, '-mx' + str(compression_level)]
    if password:
        args.append('-p' + password)
    sub = sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE)
    res = sub.wait()
    if res == 0:
        return True
    elif res == 1:
        return False
    elif res == 2:
        raise FatalError
    elif res == 7:
        raise CommandLineError
    elif res == 8:
        raise MemError
    elif res == 255:
        raise UserInterrupt
    else:
        raise SevenZipError

def test(archivepath):
    if not os.path.exists(archivepath):
        raise IOError('Could not find file/directory at path')

    args = [_SZIP_PATH, 't', archivepath, '-r']

    sub = sp.Popen(args, stdout=sp.PIPE, stdin=sp.PIPE)
    res = sub.wait()
    if res == 0:
        return True
    elif res == 1 or res == 2:
        return False
    elif res == 7:
        raise CommandLineError
    elif res == 8:
        raise MemError
    elif res == 255:
        raise UserInterrupt
    else:
        raise SevenZipError


def archivelist(archivepath):
    '''Return a list of all files in the archive.'''
    if not os.path.exists(archivepath):
        raise IOError('Could not find file/directory at path')
    args = [_SZIP_PATH, 't', archivepath, '-r']
    sub = sp.Popen(args, stdout=sp.PIPE, stdin=sp.PIPE)
    stdout = sub.communicate()[0]
    tmp = re.findall(_SZ_LIST_PTTRN, stdout)
    li = []
    for i in tmp:
        li.append(re.sub(_SZ_LIST_STRIP, '', i))
    res = sub.returncode

    if res == 0 or res == 1:
        return li
    elif res == 2:
        raise FatalError
    elif res == 7:
        raise CommandLineError
    elif res == 8:
        raise MemError
    elif res == 255:
        raise UserInterrupt
    else:
        raise SevenZipError

def move_to_archive(path, archive_name):
    import shutil
    tmp = path + '-' + archive_name
    os.rename(path, tmp)
    if not os.path.exists(path):
        os.makedirs(path)
    pack(tmp, path + '-' + archive_name + '.7z')
    shutil.rmtree(tmp)
# ---------------------------------------分割线------------------------------------------------



# 将BLS模型封装成类
# 回归
class BLSregressor:
    def __init__(self,s,C,NumFea,NumWin,NumEnhan):
        self.s = s
        self.C = C
        self.NumFea = NumFea
        self.NumEnhan = NumEnhan
        self.NumWin = NumWin

    def shrinkage(self,a,b):
        z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
        return z
        
    def tansig(self,x):
        return (2/(1+np.exp(-2*x)))-1

    def pinv(self,A,reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
    def sparse_bls(self,A,b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T,A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m,n],dtype = 'double')
        ok = np.zeros([m,n],dtype = 'double')
        uk = np.zeros([m,n],dtype = 'double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1,A.T),b)
        for i in range(itrs):
            tempc = ok - uk
            ck =  L2 + np.dot(L1,tempc)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk
    
    def fit(self,train_x,train_y):  
        train_y = train_y.reshape(-1,1)
        u = 0
        WF = list()
        for i in range(self.NumWin):
            random.seed(i+u)
            WeightFea=2*random.randn(train_x.shape[1]+1,self.NumFea)-1
            WF.append(WeightFea)
        random.seed(100)
        WeightEnhan=2*random.randn(self.NumWin*self.NumFea+1,self.NumEnhan)-1
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0],1])])
        y = np.zeros([train_x.shape[0],self.NumWin*self.NumFea])
        WFSparse = list()
        distOfMaxAndMin = np.zeros(self.NumWin)
        meanOfEachWindow = np.zeros(self.NumWin)
        for i in range(self.NumWin):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)        
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse  = self.sparse_bls(A1,H1).T
            WFSparse.append(WeightFeaSparse)
        
            T1 = H1.dot(WeightFeaSparse)
            meanOfEachWindow[i] = T1.mean()
            distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - meanOfEachWindow[i])/distOfMaxAndMin[i] 
            y[:,self.NumFea*i:self.NumFea*(i+1)] = T1
        H2 = np.hstack([y,0.1 * np.ones([y.shape[0],1])])
        T2 = H2.dot(WeightEnhan)
        T2 = self.tansig(T2)
        T3 = np.hstack([y,T2])
        WeightTop = self.pinv(T3,self.C).dot(train_y)
        self.WeightTop = WeightTop
        self.WFSparse = WFSparse
        self.meanOfEachWindow = meanOfEachWindow
        self.distOfMaxAndMin = distOfMaxAndMin
        self.WeightEnhan = WeightEnhan
        return self

    def predict(self,test_x):
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
        yy1=np.zeros([test_x.shape[0],self.NumWin*self.NumFea])
        for i in range(self.NumWin):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1  = (TT1 - self.meanOfEachWindow[i])/self.distOfMaxAndMin[i]   
            yy1[:,self.NumFea*i:self.NumFea*(i+1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1,TT2])
        NetoutTest = TT3.dot(self.WeightTop)
        NetoutTest = np.array(NetoutTest).reshape(1,-1)
        return NetoutTest
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
    def get_params(self,deep = False):
        return {
            's':self.s,
            'C':self.C,
            'NumFea':self.NumFea,
            'NumWin':self.NumWin,
            'NumEnhan':self.NumEnhan
        }

# 分类
class BLSclassification:
    def __init__(self, s, C, NumFea, NumWin, NumEnhan):
        self.s = s
        self.C = C
        self.NumFea = NumFea
        self.NumEnhan = NumEnhan
        self.NumWin = NumWin
        self.onehot = None
        self.label_size = None

    def shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def tansig(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1

    def pinv(self, A, reg):
        return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def one_hot_y(self, y):
        self.onehot = OneHotEncoder(sparse=False)
        self.onehot.fit(y.reshape(-1, 1))
        y_t = self.onehot.transform(y.reshape(-1, 1))
        return y_t

    def fit(self, train_x, train_y, sample_weight=None):
        train_y = self.one_hot_y(train_y)
        self.label_size = train_y.shape[1]
        train_y = train_y.reshape(-1, self.label_size)
        u = 0
        WF = list()
        for i in range(self.NumWin):
            random.seed(i + u)
            WeightFea = 2 * random.randn(train_x.shape[1] + 1, self.NumFea) - 1
            WF.append(WeightFea)
        WeightEnhan = 2 * random.randn(self.NumWin * self.NumFea + 1, self.NumEnhan) - 1
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        y = np.zeros([train_x.shape[0], self.NumWin * self.NumFea])
        WFSparse = list()
        distOfMaxAndMin = np.zeros(self.NumWin)
        meanOfEachWindow = np.zeros(self.NumWin)
        for i in range(self.NumWin):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse = self.sparse_bls(A1, H1).T
            WFSparse.append(WeightFeaSparse)
            T1 = H1.dot(WeightFeaSparse)
            meanOfEachWindow[i] = T1.mean()
            distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
            y[:, self.NumFea * i:self.NumFea * (i + 1)] = T1
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        T2 = H2.dot(WeightEnhan)
        T2 = self.tansig(T2)
        T3 = np.hstack([y, T2])
        WeightTop = self.pinv(T3, self.C).dot(train_y)
        self.WeightTop = WeightTop
        self.WFSparse = WFSparse
        self.meanOfEachWindow = meanOfEachWindow
        self.distOfMaxAndMin = distOfMaxAndMin
        self.WeightEnhan = WeightEnhan
        return self

    def predict(self, test_x):
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.NumWin * self.NumFea])
        for i in range(self.NumWin):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.NumFea * i:self.NumFea * (i + 1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop)
        NetoutTest = np.argmax(np.array(NetoutTest), axis=1).reshape(1, -1)[0]
        return NetoutTest

    def predict_proba(self, test_x):
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.NumWin * self.NumFea])
        for i in range(self.NumWin):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.NumFea * i:self.NumFea * (i + 1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop)
        NetoutTest = np.array(NetoutTest).reshape(-1, self.label_size)
        return NetoutTest

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=False):
        return {
            's': self.s,
            'C': self.C,
            'NumFea': self.NumFea,
            'NumWin': self.NumWin,
            'NumEnhan': self.NumEnhan
        }

# ---------------------------------------分割线------------------------------------------------


# 评估聚类的性能，内部评估指标
def calculate_cluster(X,label):
    score1 = davies_bouldin_score(X,label) # 得分越低越好
    score2 = calinski_harabasz_score(X,label) # 得分越高越好
    score3 = silhouette_score(X,label) # 越接近于1代表越好
    return [score1,score2,score3]

# ---------------------------------------分割线------------------------------------------------

# 对应model训练类的网格搜索后根据某个评估指标来寻找模型最优参数，metrics为指定的评估指标，ascending=True,最后结果为越小越好，反之，相反。
def analysiz_train_results(path,model,name,metrics,ascending=True):
    path_a = path + model + "_" + name + "_assess.csv"
    path_p = path + model + "_" + name + "_parameter.csv"
    data_a = pd.read_csv(path_a)
    data_p = pd.read_csv(path_p)
    data_a = data_a.sort_values(by=metrics,ascending=ascending)
    data_p = data_p.iloc[data_a.index]
    return data_a[:5],data_p[:5] # 返回前五个最好的结果

# ---------------------------------------分割线------------------------------------------------

# 训练调参的类，包含各个常用的模型，每个模型的参数范围只是个例子，具体任务具体设置。
class model:
    # 初始化放置结果的文件夹
    def __init__(self,path_model_csv,path_model_pic,path_best_model):
        print(os.getcwd())
        # 创建保存结果的文件夹
        self.path_model_csv = path_model_csv
        folder = os.path.exists(self.path_model_csv)
        if not folder:                   
            os.makedirs(self.path_model_csv)
        # 创建保存结果图片的文件夹
        self.path_model_pic = path_model_pic
        folder = os.path.exists(self.path_model_pic)
        if not folder:                   
            os.makedirs(self.path_model_pic)
        self.path_best_model = path_best_model
        # 创建保存模型的文件夹
        folder = os.path.exists(self.path_best_model)
        if not folder:                   
            os.makedirs(self.path_best_model)
        #self.criterion = "mse" # 保存最佳的模型的标准
        print("init")
    
    # 将参数和评估结果写入文件
    def write_csv_result(self,path_1,path_2,all_metrics,all_parameter):
        with open(path_1,"a",encoding="utf-8",newline="")as f:
            f = csv.writer(f)
            f.writerow(all_metrics)
        with open(path_2,"a",encoding="utf-8",newline="")as f:
            f = csv.writer(f)
            f.writerow(all_parameter)
    
    # 将训练集进行拆分，分成多个batch数据集
    def create_batch_data(self,batch_size):
        Train = np.c_[self.train_x,self.train_y]
        data_size = Train.shape[0]
        batch_data = []
        # 大于batch_size再进行切分。小于等于的话，就只有一个batch data
        if(data_size > batch_size):
            for i in range(int(data_size/batch_size)+1):
                start = i * batch_size
                if((i+1)*batch_size < data_size):
                    end = start + batch_size
                else:
                    end = data_size
                batch_data.append([Train[start:end,:][:,:-1],Train[start:end,:][:,-1]])
        else:
            batch_data.append([Train[:,:][:,:-1],Train[:,:][:,-1]])
        self.batch_data = batch_data

    # 数据预处理，三种不同的标准化方式
    def scaler(self,fun):
        if(fun=="MaxAbs"):
            ss = MaxAbsScaler()
            ss.fit(self.train_x)
            self.train_x = ss.transform(self.train_x)
            self.test_x = ss.transform(self.test_x)
        elif(fun=="Standard"):
            ss = StandardScaler()
            ss.fit(self.train_x)
            self.train_x = ss.transform(self.train_x)
            self.test_x = ss.transform(self.test_x)
        elif(fun=="MinMax"):
            ss = MinMaxScaler()
            ss.fit(self.train_x)
            self.train_x = ss.transform(self.train_x)
            self.test_x = ss.transform(self.test_x)
        else:
            print("Scaler error!!!")


    # 导入训练集和测试集，并且创建批量数据，导入进来的numpy的array类型数据
    def load_data(self,train_x,test_x,train_y,test_y,batch_size=2048,scaler=None):
        self.feature_size = test_x.shape[1]
        self.train_x = train_x.reshape(-1,self.feature_size)
        self.test_x = test_x.reshape(-1,self.feature_size)
        if(scaler!=None):
            self.scaler(fun=scaler)
        self.train_y = train_y.ravel()
        self.test_y = test_y.ravel()
        self.test_sample_size = test_x.shape[0]
        self.create_batch_data(batch_size=batch_size)
        print("Train X : {}, Test X : {}, Feature size : {}".format(self.train_x.shape,self.test_x.shape,self.feature_size))


    # 回归任务的评估指标，pf代表是否打印出来结果，默认不打印
    def calculate(self,y_true, y_predict, n, p ,pf=False):
        y_true = y_true.reshape(-1,1)
        y_predict = y_predict.reshape(-1,1)
        # 初始化评估结果
        mse = np.inf
        rmse = np.inf
        mae = np.inf
        r2 = np.NINF
        mad = np.inf
        mape = np.inf
        r2_adjusted = np.NINF
        rmsle = np.inf
        # try except 的原因是有时候有些结果不适合用某种评估指标
        try:
            mse = mean_squared_error(y_true, y_predict)
        except:
            pass
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_predict))
        except:
            pass
        try:
            mae = mean_absolute_error(y_true, y_predict)
        except:
            pass
        try:
            r2 = r2_score(y_true, y_predict)
        except:
            pass
        try:
            mad = median_absolute_error(y_true, y_predict)
        except:
            pass
        try:
            mape = np.mean(np.abs((y_true - y_predict) / y_true)) * 100
        except:
            pass
        try:
            if(n>p):
                r2_adjusted = 1-((1-r2)*(n-1))/(n-p-1)
        except:
            pass
        try:
            rmsle = np.sqrt(mean_squared_log_error(y_true,y_predict))
        except:
            pass
        if(pf):
            try:
                print('MSE: ', mse)
            except:
                pass
            try:
                print('RMSE: ', rmse)
            except:
                pass
            try:
                print('MAE: ', mae)
            except:
                pass
            try:
                print('R2: ', r2)
            except:
                pass
            try:
                print('MAD:', mad)
            except:
                pass
            try:
                print('MAPE:', mape)
            except:
                pass
            try:
                print('R2_Adjusted: ',r2_adjusted)
            except:
                pass
            try:
                print("RMSLE: ",rmsle)
            except:
                pass
        return mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle
    
    # Adaboost 训练调参代码
    def Ada(self,name):
        path_a = self.path_model_csv + "Ada_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Ada_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Ada_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Ada_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        n_estimators = [30,40,50,60,70]
        learning_rate = [0.1,0.5,1,1.5,2]
        loss = ["linear","square","exponential"]
        random_state = 17
        all_nb = len(n_estimators) * len(learning_rate) * len(loss)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','n_estimators','learning_rate','loss']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for n in n_estimators:
            for l in learning_rate:
                for lo in loss:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("Ada start....{}/{}".format(num,all_nb))
                        model = AdaBoostRegressor(n_estimators=n,learning_rate=l,loss=lo,random_state=random_state)
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,n,l,lo]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        num = num+1
                        print("--------------------------------")
                                                
    # KNN 训练调参代码
    def Knn(self,name):
        path_a = self.path_model_csv + "Knn_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Knn_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Knn_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Knn_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        n_neighbors = [3,5,7,9] # 默认为5
        weights = ['uniform', 'distance']
        leaf_size = [25,30,35] #默认是30
        P = [1,2] # 只在 wminkowski 和 minkowski 调
        all_nb = len(n_neighbors) * len(weights) * len(leaf_size) * len(P)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0 
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','weights','n_neighbors','leaf_size','P']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for n in n_neighbors:
            for l in leaf_size:
                for p in P:
                    for w in weights:
                        if(nums>=num):
                            num = num+1
                        else:
                            print("KNN start....{}/{}".format(num,all_nb))
                            model = KNeighborsRegressor(n_neighbors=n,leaf_size=l,p=p,weights=w)
                            model.fit(self.train_x,self.train_y)
                            pred_test = model.predict(self.test_x)
                            pred_test = pred_test.reshape(-1,1)
                            mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                            all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                            all_p = [num,w,n,l,p]
                            self.write_csv_result(path_a,path_p,all_m,all_p)
                            if(rmse < best_result):
                                joblib.dump(model,model_path)
                                with open(path_b,"a",encoding="utf-8",newline="")as f:
                                    f = csv.writer(f)
                                    f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                best_result = rmse
                                print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                            print("end....",num)
                            num = num+1

    
    #SVR训练调参代码
    def Svr(self,name):
        path_a = self.path_model_csv + "Svr_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Svr_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Svr_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Svr_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        kernel = ["rbf","linear","poly","sigmoid"]
        degree = [2,3,4,5,6,7,8,9,10,11,12]
        gamma = ["auto","scale"]
        all_nb = len(kernel) * len(degree) * len(gamma)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','kernel','degree','gamma']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for k in kernel:
            if(k=="poly"):
                for d in degree:
                    for g in gamma:
                        if(nums>=num):
                            num = num+1
                        else:
                            print("SVR start....{}/{}".format(num,all_nb))
                            model = SVR(kernel=k,degree=d,gamma=g)
                            model.fit(self.train_x,self.train_y)
                            pred_test = model.predict(self.test_x)
                            pred_test = pred_test.reshape(-1,1)
                            mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                            all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                            all_p = [num,k,d,g]
                            self.write_csv_result(path_a,path_p,all_m,all_p)
                            if(rmse < best_result):
                                joblib.dump(model,model_path)
                                with open(path_b,"a",encoding="utf-8",newline="")as f:
                                    f = csv.writer(f)
                                    f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                best_result = rmse
                                print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                            print("end....",num)
                            num = num+1
                            print("--------------------------------")
            elif(k=="rbf" or k=="sigmoid"):
                for g in gamma:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("SVR start....{}/{}".format(num,all_nb))
                        model = SVR(kernel=k,gamma=g)
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,k,"None",g]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        print("--------------------------------")
                        num = num+1
            else:
                if(nums>=num):
                    num = num+1
                else:
                    print("SVR start....{}/{}".format(num,all_nb))
                    model = SVR(kernel=k)
                    model.fit(self.train_x,self.train_y)
                    pred_test = model.predict(self.test_x)
                    pred_test = pred_test.reshape(-1,1)
                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                    all_p = [num,k,"None","None"]
                    self.write_csv_result(path_a,path_p,all_m,all_p)
                    if(rmse < best_result):
                        joblib.dump(model,model_path)
                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                            f = csv.writer(f)
                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                        best_result = rmse
                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                    print("end....",num)
                    num = num+1
                    print("--------------------------------")  

    # LinearSVR训练调参代码      
    def LinearSvr(self,name):
        path_a = self.path_model_csv + "LinearSvr_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "LinearSvr_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "LinearSvr_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "LinearSvr_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        tol = [1e-5,1e-4,1e-3]
        C = [1.0,1.5,0.5,2.0,0.01]
        loss = ["epsilon_insensitive","squared_epsilon_insensitive"]
        random_state = 17
        all_nb = len(tol) * len(C) * len(loss)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','tol','C','loss']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for t in tol:
            for c in C:
                for l in loss:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("LinearSvr start....{}/{}".format(num,all_nb))
                        model = LinearSVR(tol=t,C=c,loss=l,random_state=random_state)
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,t,c,l]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        num = num+1
                        print("--------------------------------")
                                   
    # DT训练调参代码      
    def Dt(self,name):
        path_a = self.path_model_csv + "Dt_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Dt_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Dt_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Dt_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        criterion = ["mse"]
        splitter = ["best","random"]
        min_samples_split = [2,3,4,5]
        min_samples_leaf = [1,2,3]
        random_state = 17
        all_nb = len(criterion) * len(splitter) * len(min_samples_split) * len(min_samples_leaf)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','criterion','splitter','min_samples_split','min_samples_leaf']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for c in criterion:
            for s in splitter:
                for ms in min_samples_leaf:
                    for mi in min_samples_split:
                        if(nums>=num):
                            num = num+1
                        else:
                            print("DT start....{}/{}".format(num,all_nb))
                            model = DecisionTreeRegressor(criterion=c,splitter=s,min_samples_leaf=ms,min_samples_split=mi,random_state=random_state)
                            model.fit(self.train_x,self.train_y)
                            pred_test = model.predict(self.test_x)
                            pred_test = pred_test.reshape(-1,1)
                            mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                            all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                            all_p = [num,c,s,ms,mi]
                            self.write_csv_result(path_a,path_p,all_m,all_p)
                            if(rmse < best_result):
                                joblib.dump(model,model_path)
                                with open(path_b,"a",encoding="utf-8",newline="")as f:
                                    f = csv.writer(f)
                                    f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                best_result = rmse
                                print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                            num = num+1
                            print("--------------------------------")    

    # Ext训练调参代码        
    def Ext(self,name):
        path_a = self.path_model_csv + "Ext_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Ext_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Ext_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Ext_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        n_estimators = [80,90,100,110,120]
        criterion = ["mse"]
        min_samples_split = [2,4,6,8]
        min_samples_leaf = [1,2,3,4]
        random_state = 17
        n_jobs = -1
        all_nb = len(n_estimators) * len(criterion) * len(min_samples_leaf) * len(min_samples_split)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','n_estimators','min_samples_split',' min_samples_leaf','criterion']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for mi in min_samples_split:
            for ms in min_samples_leaf:
                    for c in criterion:
                        model = ExtraTreesRegressor(n_estimators=1,warm_start=True,n_jobs=n_jobs,random_state=random_state,min_samples_leaf=ms,min_samples_split=mi,criterion=c)
                        for n in n_estimators:
                            if(nums>=num):
                                num = num+1
                            else:
                                print("EXT start....{}/{}".format(num,all_nb))
                                model.n_estimators = n
                                model.fit(self.train_x,self.train_y)
                                pred_test = model.predict(self.test_x)
                                pred_test = pred_test.reshape(-1,1)
                                mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                all_p = [num,n,mi,ms,c]
                                self.write_csv_result(path_a,path_p,all_m,all_p)
                                if(rmse < best_result):
                                    joblib.dump(model,model_path)
                                    with open(path_b,"a",encoding="utf-8",newline="")as f:
                                        f = csv.writer(f)
                                        f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                    best_result = rmse
                                    print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                num = num+1
                                print("--------------------------------")

    # GBDT训练调参代码
    def Gbdt(self,name):
        path_a = self.path_model_csv + "Gbdt_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Gbdt_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Gbdt_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Gbdt_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        random_state = 17
        n_estimators = [80,90,100,110,120]
        learning_rate = [0.05,0.1,0.15]
        subsample = [1,0.8,0.6]
        min_samples_split = [2,3,4]
        min_samples_leaf = [1,2,3]
        all_nb =  len(n_estimators) * len(learning_rate)  * len(subsample) * len(min_samples_leaf) * len(min_samples_split)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','n_estimators','learning_rate','loss','subsample','min_samples_split','max_depth','min_samples_leaf']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for l in learning_rate:
            for sub in subsample:
                for mi in min_samples_split:
                    for ms in min_samples_leaf:
                        model = GradientBoostingRegressor(warm_start=True,random_state=random_state,n_estimators=1,learning_rate=l,subsample=sub,min_samples_split=mi,min_samples_leaf=ms)
                        for n in n_estimators:
                            if(nums>=num):
                                num = num+1
                            else:
                                print("GBDT start....{}/{}".format(num,all_nb))
                                model.n_estimators = n
                                model.fit(self.train_x,self.train_y)
                                pred_test = model.predict(self.test_x)
                                pred_test = pred_test.reshape(-1,1)
                                mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                all_p = [num,n,l,sub,mi,ms]
                                self.write_csv_result(path_a,path_p,all_m,all_p)
                                if(rmse < best_result):
                                    joblib.dump(model,model_path)
                                    with open(path_b,"a",encoding="utf-8",newline="")as f:
                                        f = csv.writer(f)
                                        f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                    best_result = rmse
                                    print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                print("end....",num)
                                num = num+1
                                print("--------------------------------")

    # RF训练模型代码
    def Rf(self,name):
        path_a = self.path_model_csv + "Rf_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Rf_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Rf_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Rf_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        n_estimators = [80,90,100,110,120]
        criterion = ["mse","mse"]
        min_samples_split = [2,3,4]
        min_samples_leaf = [1,2,3]
        random_state = 17
        n_jobs = -1
        all_nb =  len(n_estimators) * len(criterion) * len(min_samples_leaf) * len(min_samples_split)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','n_estimators','min_samples_split','min_samples_leaf','criterion']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for mi in min_samples_split:
            for ms in min_samples_leaf:
                for c in criterion:
                    model = RandomForestRegressor(n_jobs=n_jobs,random_state=random_state,n_estimators=1,warm_start=True,min_samples_leaf=ms,min_samples_split=mi,criterion=c)
                    for n in n_estimators:
                        if(nums>=num):
                            num = num+1
                        else:
                            print("RF start....{}/{}".format(num,all_nb))
                            model.n_estimators = n
                            model.fit(self.train_x,self.train_y)
                            pred_test = model.predict(self.test_x)
                            pred_test = pred_test.reshape(-1,1)
                            mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                            all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                            all_p = [num,n,mi,ms,c]
                            self.write_csv_result(path_a,path_p,all_m,all_p)
                            if(rmse < best_result):
                                joblib.dump(model,model_path)
                                with open(path_b,"a",encoding="utf-8",newline="")as f:
                                    f = csv.writer(f)
                                    f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                best_result = rmse
                                print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                            num = num+1
                            print("--------------------------------")

    # XGBoot调参训练代码
    def Xgb(self,name):
        path_a = self.path_model_csv + "Xgb_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Xgb_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Xgb_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Xgb_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围

        eta = [ 0.1, 0.2, 0.3, 0.4, 0.5]
        subsample = [1,0.8,0.6]
        reg_lambda = [0,0.5,1]
        reg_alpha = [0,0.5,1]
        gamma = [0,0.5,1]
        all_nb = len(eta)*len(subsample)*len(reg_lambda)*len(reg_alpha)*len(gamma)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','eta','subsample','reg_lambda','reg_alpha','gamma']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for e in eta:
            for s in subsample:
                for rl in reg_lambda:
                    for ra in reg_alpha:
                        for g in gamma:
                            if(nums>=num):
                                num = num+1
                            else:
                                print("XGB train...{}/{}".format(num,all_nb))
                                model = xgb.XGBRegressor(gamma=g, reg_alpha=ra, reg_lambda=rl, subsample=s, eta=e)
                                model.fit(self.train_x,self.train_y)
                                pred_test = model.predict(self.test_x)
                                pred_test = pred_test.reshape(-1,1)
                                mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                all_p = [num,e,s,rl,ra,g]
                                self.write_csv_result(path_a,path_p,all_m,all_p)
                                if(rmse < best_result):
                                    joblib.dump(model,model_path)
                                    with open(path_b,"a",encoding="utf-8",newline="")as f:
                                        f = csv.writer(f)
                                        f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                    best_result = rmse
                                    print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                print("end....",num)
                                num = num+1
                                print("--------------------------------")

    # Catboost调参训练代码
    def Cat(self,name):
        path_a = self.path_model_csv + "Cat_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Cat_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Cat_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Cat_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围

        learning_rate=[0.003,0.01,0.03,0.05,0.07,0.09,0.1]
        iterations = [1300,1200,1100,1000,900,800]
        l2_leaf_reg = [1,2,3,4,5]
        all_nb = len(learning_rate)*len(iterations)*len(l2_leaf_reg)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','learning_rate','iterations','l2_leaf_reg']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for l in learning_rate:
            for i in iterations:
                for l2 in l2_leaf_reg:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("CAT train...{}/{}".format(num,all_nb))
                        model = cb.CatBoostRegressor(learning_rate=l,iterations=i,l2_leaf_reg=l2,logging_level='Silent',random_seed=17)
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,l,i,l2]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        num = num+1
                        print("--------------------------------")


    # Lgboost训练调参代码
    def Lgb(self,name):
        path_a = self.path_model_csv + "Lgb_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Lgb_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Lgb_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Lgb_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        boosting_type = ["gbdt","dart","goss"] #,"rf"
        n_estimators = [50,75,100,125,150,175,200]
        learning_rate = [0.01,0.05,0.1,0.15,0.2]
        random_state = 17
        colsample_bytree = [1,0.8]
        lambda_l1 = [0,1,2]
        lambda_l2 = [0,1,2]
        all_nb = len(boosting_type)*len(n_estimators)*len(learning_rate)*len(lambda_l1)*len(lambda_l2)*len(colsample_bytree)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','boosting_type','n_estimators','learning_rate','colsample_bytree','lambda_l1','lambda_l2']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for b in boosting_type:
            for n in n_estimators:
                for l in learning_rate:
                    for c in colsample_bytree:
                        for rl1 in lambda_l1:
                            for rl2 in lambda_l2:
                                if(nums>=num):
                                    num = num+1
                                else:
                                    print("LGB train...{}/{}".format(num,all_nb))
                                    model = lgb.LGBMRegressor(lambda_l1=rl1,lambda_l2=rl2,colsample_bytree=c,learning_rate=l,n_estimators=n,boosting_type=b,objective='regression',random_state=random_state)
                                    model.fit(self.train_x,self.train_y)
                                    pred_test = model.predict(self.test_x)
                                    pred_test = pred_test.reshape(-1,1)
                                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                    all_p = [num,b,n,l,c,rl1,rl2]
                                    self.write_csv_result(path_a,path_p,all_m,all_p)
                                    if(rmse < best_result):
                                        joblib.dump(model,model_path)
                                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                                            f = csv.writer(f)
                                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                        best_result = rmse
                                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                    num = num+1
                                    print("--------------------------------")

    # Bagging调参训练代码
    def Bagging(self,name):
        path_a = self.path_model_csv + "Bagging_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Bagging_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Bagging_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Bagging_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        n_estimators = [10+(5*i) for i in range(200)]
        max_samples = [0.7,0.8,0.9,1.0]
        max_features = [0.7,0.8,0.9,1.0]
        all_len = len(n_estimators) * len(max_samples) * len(max_features)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','n_estimators','max_samples','max_features']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for mf in max_features:
            for ma in max_samples:
                model = BaggingRegressor(warm_start=True,n_estimators=1, random_state=17,max_samples=ma,max_features=mf)
                for n in n_estimators:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("Bagging train...{}/{}".format(num,all_len))
                        model.n_estimators = n
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,ma,mf,n]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        num = num+1
                        print("--------------------------------") 

    #Bagging-BLS调参训练代码
    def Bagging_Bls(self,name):
        path_a = self.path_model_csv + "Bagging_Bls_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Bagging_Bls_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Bagging_Bls_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Bagging_Bls_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        s = 0.4
        c = 2**-30
        nf = 30
        nw = 5
        ne = 5
        n_estimators = [10+(5*i) for i in range(200)]
        max_samples = [0.7,0.8,0.9,1.0]
        max_features = [0.7,0.8,0.9,1.0]
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','s','C','NumFea','NumWin','NumEnhan','n_estimators','max_samples','max_features']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        all_len = len(n_estimators) * len(max_samples) * len(max_features)
        for mf in max_features:
            for ma in max_samples:
                model = BaggingRegressor(base_estimator=BLSregressor(s=s,C=c,NumFea=nf,NumWin=nw,NumEnhan=ne),warm_start=True,n_estimators=1, random_state=17,max_samples=ma,max_features=mf)
                for n in n_estimators:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("Bagging_BLS train...{}/{}".format(num,all_len))
                        model.n_estimators = n
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,s,c,nf,nw,ne,ma,mf,n]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        num = num+1
                        print("--------------------------------")   
    
    # BLS训练调参代码
    def Bls(self,name):
        path_a = self.path_model_csv + "Bls_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Bls_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Bls_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Bls_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        NumFea = [i for i in range(2,40,4)]
        NumWin = [i for i in range(5,40,5)]
        NumEnhan = [i for i in range(5,60,10)]
        S = [0.4,0.6,0.8,1,1.2,4]
        C = [2**-30,2**-10,2**-20,2**-40,1**-30]
        all_nb = len(NumFea)*len(NumWin)*len(S)*len(C)*len(NumEnhan)
        num=1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','s','C','NumFea','NumWin','NumEnhan']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for nf in NumFea:
            for nw in NumWin:
                for s in S:
                    for c in C:
                        for ne in NumEnhan:
                            if(nums>=num):
                                num = num+1
                            else:
                                print("BLS train...{}/{}".format(num,all_nb))
                                model = BLSregressor(s=s, C=c, NumFea=nf, NumWin=nw, NumEnhan=ne)
                                model.fit(self.train_x,self.train_y)
                                pred_test = model.predict(self.test_x)
                                pred_test = pred_test.reshape(-1,1)
                                mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                all_p = [num,s,c,nf,nw,ne]
                                self.write_csv_result(path_a,path_p,all_m,all_p)
                                if(rmse < best_result):
                                    joblib.dump(model,model_path)
                                    with open(path_b,"a",encoding="utf-8",newline="")as f:
                                        f = csv.writer(f)
                                        f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                    best_result = rmse
                                    print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                print("end....",num)
                                num = num+1
                                print("--------------------------------")   
         
    # MLP调参训练代码
    def Mlp(self,name):
        path_a = self.path_model_csv + "Mlp_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Mlp_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Mlp_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Mlp_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        max_iter = [2000,5000,8000,10000,12000,15000,20000]
        tol = [1e-3,2e-3,1e-4,1e-2]
        learning_rate_init = [1e-2,1e-3,1e-4]
        hidden_layer_sizes = list(combinations([64,32,16,8,4], 3))
        activation = ["identity","logistic","tanh","relu"]
        solver = ["lbfgs","sgd","adam"]
        all_nb = len(max_iter) * len(tol) * len(learning_rate_init) * len(hidden_layer_sizes) * len(activation) * len(solver)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0 
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','max_iter','tol','learning_rate_init','hidden_layer_sizes','activation','solver']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for m in max_iter:
            for t in tol:
                for l in learning_rate_init:
                    for hd in hidden_layer_sizes:
                        for a in activation:
                            for s in solver:
                                if(nums>=num):
                                    num = num+1
                                else:
                                    print("MLP start....{}/{}".format(num,all_nb))
                                    model = MLPRegressor(hidden_layer_sizes=hd, activation=a,
                                                     solver=s, alpha=0.0001,
                                                     batch_size='auto', learning_rate="constant",
                                                     learning_rate_init=l,
                                                     power_t=0.5, max_iter=m,tol=t)
                                    model.fit(self.train_x,self.train_y)
                                    pred_test = model.predict(self.test_x)
                                    pred_test = pred_test.reshape(-1,1)
                                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                    all_p = [num,m,t,l,hd,s,a]
                                    self.write_csv_result(path_a,path_p,all_m,all_p)
                                    if(rmse < best_result):
                                        joblib.dump(model,model_path)
                                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                                            f = csv.writer(f)
                                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                        best_result = rmse
                                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                    print("end....",num)
                                    num = num+1
                                    print("--------------------------------")  

    # 一些 Linear model 调参训练代码
    def Lm(self,name):
        path_a = self.path_model_csv + "Lm_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Lm_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Lm_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Lm_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        Model=['Ridge','Lasso','ElasticNet']
        alpha=[0,0.001,0.01,0.1,1,10,100]
        max_iter=[10,100,500,1000,2000,3000,10000]
        tol=[1e-3,3e-3,2e-3,1e-4,4e-3]
        solver=["auto",'svd','cholesky','sparse_cg','sag']
        random_state=17
        precompute=[True,False]
        selection=['cyclic']
        l1_ratio=[0.1,0.3,0.5,0.7,0.9]
        all_nb = len(alpha)*len(max_iter)*len(tol)*len(solver)+len(alpha)*(len(max_iter)-1)*len(tol)*len(solver)*len(precompute)*len(selection)+len(alpha)*len(max_iter)*len(tol)*len(solver)*len(precompute)*len(selection)*len(l1_ratio)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','Model','alpha','max_iter','tol','solver','precompute','selection','l1_ratio']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for mo in Model:
            for a in alpha:
                for mi in max_iter:
                    for to in tol:
                        for so in solver:
                            if(mo=='Ridge'):
                                if(nums>=num):
                                    num = num+1
                                else:
                                    print("Ridge start....{}/{}".format(num,all_nb))
                                    model = Ridge(alpha=a, max_iter=mi, tol=to, solver=so,random_state=random_state)
                                    model.fit(self.train_x,self.train_y)
                                    pred_test = model.predict(self.test_x)
                                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                    all_p = [num,mo,a,mi,to,so,"None","None"]
                                    self.write_csv_result(path_a,path_p,all_m,all_p)
                                    if(rmse < best_result):
                                        joblib.dump(model,model_path)
                                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                                            f = csv.writer(f)
                                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                        best_result = rmse
                                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                    print('end...',num)
                                    num = num+1
                            else:
                                for pr in precompute:
                                    for se in selection:
                                        if(mo=='Lasso'):
                                            if(nums>=num):
                                                num = num+1
                                            else:
                                                print("Lasso start....{}/{}".format(num,all_nb))
                                                model = Lasso(alpha=a, max_iter=mi, tol=to,random_state=random_state,precompute=pr,selection=se)
                                                model.fit(self.train_x,self.train_y)
                                                pred_test = model.predict(self.test_x)
                                                mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                                all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                                all_p = [num,mo,a,mi,to,so,pr,se,"None"]
                                                self.write_csv_result(path_a,path_p,all_m,all_p)
                                                if(rmse < best_result):
                                                    joblib.dump(model,model_path)
                                                    with open(path_b,"a",encoding="utf-8",newline="")as f:
                                                        f = csv.writer(f)
                                                        f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                                    best_result = rmse
                                                    print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                                print('end...',num)
                                                num = num+1
                                        elif(mo=='ElasticNet'):
                                            so=''
                                            for l1 in l1_ratio:
                                                if(nums>=num):
                                                    num = num+1
                                                else:
                                                    print("ElasticNet start....{}/{}".format(num,all_nb))
                                                    model = ElasticNet(alpha=a, max_iter=mi, tol=to,random_state=random_state,precompute=pr,selection=se,l1_ratio=l1)
                                                    model.fit(self.train_x,self.train_y)
                                                    pred_test = model.predict(self.test_x)
                                                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                                    all_p = [num,mo,a,mi,to,so,pr,se,l1]
                                                    self.write_csv_result(path_a,path_p,all_m,all_p)
                                                    if(rmse < best_result):
                                                        joblib.dump(model,model_path)
                                                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                                                            f = csv.writer(f)
                                                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                                        best_result = rmse
                                                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                                    print('end...',num)
                                                    num = num+1

    # LR SGD调参训练代码
    def Lr_Sgd(self,name):
        path_a = self.path_model_csv + "Lr_Sgd_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Lr_Sgd_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Lr_Sgd_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Lr_Sgd_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        loss = ["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"]
        penalty = ["l2","l1","elasticnet"]
        alpha = [0.0001,0.001,0.0005,0.01]
        l1_ratio = [0.15,0.1,0.2,0.3,0.01]
        tol = [1e-3,1e-2,1e-4]
        learning_rate = ["constant","optimal","invscaling","adaptive"]
        eta0 = [0.01,0.015,0.005]
        power_t = [0.25,0.2,0.3]
        random_state = 17
        all_nb = len(loss) * len(penalty) * len(alpha) * len(l1_ratio) * len(tol) * len(learning_rate) * len(eta0) * len(power_t)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','loss','penalty','alpha','l1_ratio','tol','learning_rate','eta0','power_t']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for l in loss:
            for p in penalty:
                for a in alpha:
                    for t in tol:
                        for lr in learning_rate:
                            for e in eta0:
                                for pt in power_t:
                                    if(p=="elasticnet"):
                                        for l1 in l1_ratio: # Only used if penalty is ‘elasticnet’.
                                            if(nums>=num):
                                                num = num+1
                                            else:
                                                print("SGDRegressor start....{}/{}".format(num,all_nb))
                                                model = SGDRegressor(random_state=random_state,warm_start=True,l1_ratio=l1,loss=l,penalty=p,alpha=a,tol=t,learning_rate=lr,eta0=e,power_t=pt)
                                                for batch in self.batch_data:
                                                    model.partial_fit(batch[0],batch[1])
                                                pred_test = model.predict(self.test_x)
                                                mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                                all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                                all_p = [num,l,p,a,l1,t,lr,e,pt]
                                                self.write_csv_result(path_a,path_p,all_m,all_p)
                                                if(rmse < best_result):
                                                    joblib.dump(model,model_path)
                                                    with open(path_b,"a",encoding="utf-8",newline="")as f:
                                                        f = csv.writer(f)
                                                        f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                                    best_result = rmse
                                                    print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                                num = num+1
                                    else:
                                        if(nums>=num):
                                            num = num+1
                                        else:
                                            print("SGDRegressor start....{}/{}".format(num,all_nb))
                                            model = SGDRegressor(random_state=random_state,warm_start=True,loss=l,penalty=p,alpha=a,tol=t,learning_rate=lr,eta0=e,power_t=pt)
                                            for batch in self.batch_data:
                                                model.partial_fit(batch[0],batch[1])
                                            pred_test = model.predict(self.test_x)
                                            mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                            all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                            all_p = [num,l,p,a,"None",t,lr,e,pt]
                                            self.write_csv_result(path_a,path_p,all_m,all_p)
                                            if(rmse < best_result):
                                                joblib.dump(model,model_path)
                                                with open(path_b,"a",encoding="utf-8",newline="")as f:
                                                    f = csv.writer(f)
                                                    f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                                best_result = rmse
                                                print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                            num = num+1

    # PAR调参训练代码
    def Par(self,name):
        path_a = self.path_model_csv + "Par_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Par_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Par_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Par_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        C = [1.0,0.5,1.5,2.0]
        tol = [1e-3,1e-2,1e-4]
        random_state = 17
        all_nb = len(C) * len(tol)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','C','tol']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for c in C:
            for t in tol:
                if(nums>=num):
                    num = num+1
                else:
                    print("PassiveAggressiveRegressor start....{}/{}".format(num,all_nb))
                    model = PassiveAggressiveRegressor(C=c,tol=t,warm_start=True,random_state=random_state)
                    for batch in self.batch_data:
                        model.partial_fit(batch[0],batch[1])
                    pred_test = model.predict(self.test_x)
                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                    all_p = [num,c,t]
                    self.write_csv_result(path_a,path_p,all_m,all_p)
                    if(rmse < best_result):
                        joblib.dump(model,model_path)
                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                            f = csv.writer(f)
                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                        best_result = rmse
                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                    num = num+1

    # BR调参训练代码
    def BR(self,name):
        path_a = self.path_model_csv + "BR_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "BR_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "BR_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "BR_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        n_iter = [100,200,300,400,500]
        tol = [1e-3,2e-3,1e-4,1e-2]
        alpha_1 = [1e-6,1e-4,1e-5]
        alpha_2 = [1e-6,1e-4,1e-5]
        lambda_1 = [1e-6,1e-4,1e-5]
        lambda_2 = [1e-6,1e-4,1e-5]
        normalize = [True,False]
        all_nb = len(n_iter) * len(tol) * len(alpha_1) * len(alpha_2) * len(lambda_1) * len(lambda_2) * len(normalize)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0 
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','n_iter','tol','alpha_1','alpha_2','lambda_1','lambda_2','normalize']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for n in n_iter:
            for t in tol:
                for a1 in alpha_1:
                    for a2 in alpha_2:
                        for l1 in lambda_1:
                            for l2 in lambda_2:
                                for nm in normalize:
                                    if(nums>=num):
                                        num = num+1
                                    else:
                                        print("BR start....{}/{}".format(num,all_nb))
                                        model = BayesianRidge(n_iter=n,tol=t,alpha_1=a1,alpha_2=a2,lambda_1=l1,lambda_2=l2,normalize=nm)
                                        model.fit(self.train_x,self.train_y)
                                        pred_test = model.predict(self.test_x)
                                        pred_test = pred_test.reshape(-1,1)
                                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                                        all_p = [num,n,t,a1,a2,l1,l2,nm]
                                        self.write_csv_result(path_a,path_p,all_m,all_p)
                                        if(rmse < best_result):
                                            joblib.dump(model,model_path)
                                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                                f = csv.writer(f)
                                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                                            best_result = rmse
                                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                                        print("end....",num)
                                        num = num+1
                                        print("--------------------------------") 
    
    # GP调参训练代码
    def GP(self,name):
        path_a = self.path_model_csv + "GP_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "GP_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "GP_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "GP_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        alpha = [1e-10,1e-9,1e-11,1e-8]
        normalize_y = [True,False]
        all_nb = len(alpha) * len(normalize_y)
        random_state = 17
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0 
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','alpha','normalize_y']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for a in alpha:
            for n in normalize_y:
                if(nums>=num):
                    num = num+1
                else:
                    print("GP start....{}/{}".format(num,all_nb))
                    model = GaussianProcessRegressor(alpha=a,normalize_y=n,random_state=random_state)
                    model.fit(self.train_x,self.train_y)
                    pred_test = model.predict(self.test_x)
                    pred_test = pred_test.reshape(-1,1)
                    mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                    all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                    all_p = [num,a,n]
                    self.write_csv_result(path_a,path_p,all_m,all_p)
                    if(rmse < best_result):
                        joblib.dump(model,model_path)
                        with open(path_b,"a",encoding="utf-8",newline="")as f:
                            f = csv.writer(f)
                            f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                        best_result = rmse
                        print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                    print("end....",num)
                    num = num+1
                    print("--------------------------------") 

    # LogR调参训练代码
    def LogR(self,name):
        path_a = self.path_model_csv + "LogR_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "LogR_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "LogR_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "LogR_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        penalty = ["l1","l2"]
        tol = [1e-3,1e-4,1e-5]
        C = [0.5,1,1.5,2.0]
        all_nb = len(penalty) * len(tol) * len(C)
        random_state = 17
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        num = 1
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0 
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','penaly','tol','C']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for p in penalty:
            for t in tol:
                for c in C:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("LogR start....{}/{}".format(num,all_nb))
                        model = LogisticRegression(penalty=p,tol=t,C=c,warm_start=True,random_state=random_state,n_jobs=-1)
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,p,t,c]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        num = num+1
                        print("--------------------------------") 

    def Lr(self,name):
        path_a = self.path_model_csv + "Lr_" + name + "_" + "assess.csv"
        path_p = self.path_model_csv + "Lr_" + name + "_" + "parameter.csv"
        path_b = self.path_model_csv + "Lr_" + name + "_" + "best.csv"
        model_path = self.path_best_model + "Lr_" + name + "_" + "best.m"
        # 人工设置网格搜索的范围
        fit_intercept = [True,False]
        normailze = [True,False]
        positive = [True,False]
        all_nb = len(fit_intercept) * len(normailze) * len(positive)
        num = 1
        # 用于重启训练模型，提高效率，不重复跑相同的实验
        if(os.path.exists(path_a)):
            data = pd.read_csv(path_a)
            if(data.shape[0]>1):
                nums = int(data.values[-1,0])
            else:
                nums = 0
        else:
            nums = 0
            col_a = ['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle']
            col_p = ['num','fit_intercept','normailze','positive']
            self.write_csv_result(path_a,path_p,col_a,col_p)
        # 用于保存最好的模型
        if(os.path.exists(path_b)):
            best_results = pd.read_csv(path_b)
            if(best_results.shape[0]>1):
                best_result = best_results["rmse"].values[-1]
            else:
                best_result = 10*10**30
        else:
            with open(path_b,"a",encoding="utf-8",newline="")as f:
                f = csv.writer(f)
                f.writerow(['num','mse','rmse','mae','r2','mad','mape','r2_adjusted','rmsle'])
            best_result = 10*10**30
        # 网格搜索
        for ft in fit_intercept:
            for n in normailze:
                for p in positive:
                    if(nums>=num):
                        num = num+1
                    else:
                        print("LR start....{}/{}".format(num,all_nb))
                        model = LinearRegression(fit_intercept=ft,normalize=n,positive=p,n_jobs=-1)
                        model.fit(self.train_x,self.train_y)
                        pred_test = model.predict(self.test_x)
                        pred_test = pred_test.reshape(-1,1)
                        mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle = self.calculate(self.test_y,pred_test,self.test_sample_size,self.feature_size)
                        all_m = [num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle]
                        all_p = [num,ft,n,p]
                        self.write_csv_result(path_a,path_p,all_m,all_p)
                        if(rmse < best_result):
                            joblib.dump(model,model_path)
                            with open(path_b,"a",encoding="utf-8",newline="")as f:
                                f = csv.writer(f)
                                f.writerow([num,mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle])
                            best_result = rmse
                            print("current best result, mse:{},mae:{},rmse:{},mad:{},r2:{},mape:{},r2 adjusted:{},rmsle:{}".format(mse,mad,rmse,mad,r2,mape,r2_adjusted,rmsle))
                        print("end....",num)
                        num = num+1
                        print("--------------------------------")


    # 根据输入的model_name选择启动调参训练的模型。name是保存结果的文件的后缀，用于区分不同的实验。                                     
    def run(self,model_name,name):
        # 1) Adaboost
        if(model_name=="Ada"):
            self.Ada(name)
            return
        # 2) KNN
        if(model_name=="Knn"):
            self.Knn(name)
            return
        # 3) SVR
        if(model_name=="Svr"):
            self.Svr(name)
            return
        # 4) DT
        if(model_name=="Dt"):
            self.Dt(name)
            return
        # 5) EXT
        if(model_name=="Ext"):
            self.Ext(name)
            return
        # 6) GBDT
        if(model_name=="Gbdt"):
            self.Gbdt(name)
            return
        # 7) RF
        if(model_name=="Rf"):
            self.Rf(name)
            return
        # 8) XGBoost
        if(model_name=="Xgb"):
            self.Xgb(name)
            return
        # 9) CatBoost
        if(model_name=="Cat"):
            self.Cat(name)
            return
        # 10) Lightbgm
        if(model_name=="Lgb"):
            self.Lgb(name)
            return
        # 11) MLP
        if(model_name=="Mlp"):
            self.Mlp(name)
            return
        # 12) Bagging BLS
        if(model_name=="Bagging_Bls"):
            self.Bagging_Bls(name)
            return
        # 13) BLS
        if(model_name=="Bls"):
            self.Bls(name)
            return
        # 14) Bagging
        if(model_name=="Bagging"):
            self.Bagging(name)
            return
        # 15) Linear SVR
        if(model_name=="LinearSvr"):
            self.LinearSvr(name)
            return
        # 16) Linear Regression
        if(model_name=="Lr"):
            self.Lr(name)
            return
        # 17) LR SGD
        if(model_name=="Lr_Sgd"):
            self.Lr_Sgd(name)
            return
        # 18) Passive Aggressive Regression
        if(model_name=="Par"):
            self.Par(name)
            return
        # 19) Bayesian Ridge
        if(model_name=="BR"):
            self.BR(name)
            return
        # 20) Gaussian Process Regression
        if(model_name=="GP"):
            self.GP(name)
            return
        # 21) Logistic Regression
        if(model_name=="LogR"):
            self.LogR(name)
            return
        # 22) Linear model
        if(model_name=="Lm"):
            self.Lm(name)
            return
# ---------------------------------------分割线------------------------------------------------


# 支持中文
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据分析类
class analysis:
    # 初始化放置分析结果的文件夹
    def __init__(self,path_pic):
        print(os.getcwd())
        # 创建保存分析结果的文件夹
        self.path_pic = path_pic
        folder = os.path.exists(self.path_pic)
        if not folder:
            os.makedirs(self.path_pic)

    # 导入数据集
    def load_data(self,data,y_col="Y"):
        self.data = data
        self.y_col = y_col


    # 相关性分析，三种相关性分析方法
    def correlation(self):
        path_fold = self.path_pic + "correlation/"
        # 计算相关性
        corr_p = self.data.corr(method='pearson')
        corr_k = self.data.corr(method='kendall')
        corr_s = self.data.corr(method='spearman')

        # 相关性热力图
        # 原始数值
        plt.figure(figsize=(25, 20))
        sns.heatmap(corr_p,  cmap="rainbow")
        plt.title("Pearson raw")
        path = path_fold + "Pearson_raw_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(25, 20))
        sns.heatmap(corr_k, cmap="rainbow")
        plt.title("Kendall raw")
        path = path_fold + "Kendall_raw_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(25, 20))
        sns.heatmap(corr_s, cmap="rainbow")
        plt.title("Spearman raw")
        path = path_fold + "Spearman_raw_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        # 取了绝对值
        plt.figure(figsize=(25, 20))
        sns.heatmap(np.abs(corr_p), cmap="rainbow")
        plt.title("Pearson abs")
        path = path_fold + "Pearson_abs_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(25, 20))
        sns.heatmap(np.abs(corr_k), cmap="rainbow")
        plt.title("Kendall abs")
        path = path_fold + "Kendall_abs_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(25, 20))
        sns.heatmap(np.abs(corr_s), cmap="rainbow")
        plt.title("Spearman abs")
        path = path_fold + "Spearman_abs_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()

        # 相关性分布图
        # 原始数值
        plt.figure(figsize=(10, 10))
        sns.distplot(corr_p[self.y_col].sort_values()[:-1].values)
        plt.title("Pearson raw")
        path = path_fold + "Pearson_raw_dist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(10, 10))
        sns.distplot(corr_k[self.y_col].sort_values()[:-1].values)
        plt.title("Kendall raw")
        path = path_fold + "Kendall_raw_dist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(10, 10))
        sns.distplot(corr_s[self.y_col].sort_values()[:-1].values)
        plt.title("Spearman raw")
        path = path_fold + "Spearman_raw_dist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        # 取了绝对值
        plt.figure(figsize=(10, 10))
        sns.distplot(np.abs(corr_p[self.y_col].sort_values()[:-1].values))
        plt.title("Pearson abs")
        path = path_fold + "Pearson_abs_dist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(10, 10))
        sns.distplot(np.abs(corr_k[self.y_col].sort_values()[:-1].values))
        plt.title("Kendall abs")
        path = path_fold + "Kendall_abs_dist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(10, 10))
        sns.distplot(np.abs(corr_s[self.y_col].sort_values()[:-1].values))
        plt.title("Spearman abs")
        path = path_fold + "Spearman_abs_dist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()









