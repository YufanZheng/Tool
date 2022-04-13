# Tool：提高实验效率的时序预测工具包（测试版）
Tool Kit for Improving Experimental Efficiency (Beta)


## 欢迎大家使用和指正！！！ （Welcome to use and correct！！！）   
## Email：zhjpre@gmail.com

## 此工具只用于提高工作效率，请在调用工具包的同时，注重个人理论知识的积累，切勿成为一个调包侠！！！

## 每个模型的参数范围需自行针对每个任务进行合理的设定。

### Sample：

train_sample.py：调用训练的示例代码（Sample code to invoke the training）；

data.csv：示例代码所使用的数据集（The data set used by the sample code）；

注：train_sample.py 调用的是最新版本的包（The call is the latest version of the package）；


### 函数或类的功能简介（A brief description of the function or class）

| Name      | Description | Type     |   Other  |
| :----:        |    :----:   |    :----:   |    :----:   |
| file_name      |  查找指定文件夹名下的所有文件的名称  |  function  |    |
| mkdir   |  创建文件夹 | function  |    |
| create_time_seq   |  创建时间序列数据集  |  function  |     |
| BLSregressor   |  封装好的 Bord Learning System 的回归模型类  | class  |     |
| BLSclassification   |  封装好的 Bord Learning System 的分类模型类  | class  |     |
| model   |  训练传统机器学习和集成学习模型的类  |  class |   各个模型的网格搜索范围要根据具体任务具体设置   |
| analysis   |  数据分析的类  |  class |      |
| calculate_cluster   |  评估聚类算法的内部评估指标  |  function |      |
|  analysiz_train_results  |  model类对应寻找最优结果的工具  |  function |      |


### 更正日志：

2021-04-28 00:00 --- 加入21个sklearn回归模型训练网格调参代码；

2021-04-29 10:01 --- 加入寻找文件名、创建文件夹、创建时间序列数据 这三个函数；

2021-05-06 15:41 --- 补充修改一些漏洞（忘了导包、变量错误等等），基本功能没有改变；

2021-05-07 11:34 --- 发现重启训练那里写入读取有问题，进行了修正，基本功能没有改变；

2021-05-08 12:33 --- 加入自动保存最好结果的模型的代码；

2021-05-11 16:33 --- 改正了训练RF和LogR的代码，改进了重启训练的代码，基本功能没有改变；

2021-05-12 20:24 --- 改正了训练Lgb和Xgb的代码，基本功能没有改变；

2021-05-14 15:22 --- 加入了使用最小二乘法去优化的线性回归模型，最基本的线性回归模型；

2021-05-17 12:34 --- 对评估函数的初始化进行了修正，加入了一个初步的数据分析类analysis；

2021-06-21 19:09 --- 加入了BLS的分类、聚类算法的内部评估指标函数、7z压缩函数，model类训练后对应寻找最优参数的函数；

2021-08-10 13:29 --- 对KNN、RF、EXT、Ada、GBDT、LGB、XGB、CAT和LinearSVR的参数范围（缩小到默认参数上下波动）进行调整；

2021-08-12 14:18 --- 修复LR模型训练函数中的写入错误；

2021-09-20 15:21 --- 修复参数设置错误导致出现报错的情况；

2021-12-14 19:17 --- 修改了Logistics的参数；增加了15个新的回归问题的评估指标；




