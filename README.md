# Tool：提高实验效率的工具包（测试版）
Tool Kit for Improving Experimental Efficiency (Beta)


## 欢迎大家使用和指正！！！ （Welcome to use and correct！！！）   
## Email：zhjpre@gmail.com


### Sample：

train_sample.py：调用训练的示例代码（Sample code to invoke the training）；

data.csv：示例代码所使用的数据集（The data set used by the sample code）；

注：train_sample.py 调用的是最新版本的包（The call is the latest version of the package）；


### 函数或类的功能简介（A brief description of the function or class）

| Name      | Description | Type     |
| :----:        |    :----:   |    :----:   |
| file_name      |  查找指定文件夹名下的所有文件的名称  |  function  |
| mkdir   |  创建文件夹 | function  |
| create_time_seq   |  创建时间序列数据集  |  function  |
| BLSregressor   |  封装好的 Bord Learning System 的模型类  | Class  |
| model   |  训练传统机器学习和集成学习模型的类  |  Class |


### 更正日志：

2021-04-28 00:00 --- 加入21个sklearn回归模型训练网格调参代码；

2021-04-29 10:01 --- 加入寻找文件名、创建文件夹、创建时间序列数据 这三个函数；

2021-05-06 15:41 --- 补充修改一些漏洞（忘了导包、变量错误等等），基本功能没有改变；

2021-05-07 11:34 --- 发现重启训练那里写入读取有问题，进行了修正，基本功能没有改变；

2021-05-08 12:33 --- 加入自动保存最好结果的模型的代码；

2021-05-11 16:33 --- 改正了训练RF和LogR的代码，改进了重启训练的代码，基本功能没有改变；

2021-05-12 20:24 --- 改正了训练Lgb和Xgb的代码，基本功能没有改变；

2021-05-14 15:22 --- 加入了使用最小二乘法去优化的线性回归模型，最基本的线性回归模型；

2021-05-17 00:23 --- 对评估函数的初始化进行了修正；






