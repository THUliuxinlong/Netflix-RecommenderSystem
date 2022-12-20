# Recommender System

## 数据集说明  
数据集链接：https://drive.google.com/drive/folders/1gRss2XCEi94xhEcH96Bf_KYV0SciF_iz?usp=sharing

数据集采用Netflix推荐竞赛的一个子集，包含10000个用户和10000个电影，具体的文件格式如下

- 用户列表 users.txt: 文件有 10000 行，每行一个整数，表示用户的 id，文件对应本次 Project 的所有用户。

- 训练集 netflix_train.txt: 文件包含 689 万条用户打分，每行为一次打分，对应的格式为: 用户 id 电影 id 分数 打分日期 其中用户 id 均出现在 users.txt 中，电影 id 为 1 到 10000 的整数。各项之间用空格分开 

- 测试集 netflix_test.txt: 文件包含约 172 万条用户打分，格式与训练集相同。  

- movie_titles.txt: 电影名称

## 代码说明  
- CF.py: 协同过滤算法的python实现 

- MD.py: 矩阵分解算法的python实现  

## 推荐系统总结  
推荐系统的核心问题是确定用户-内容矩阵(Utility Matrix) 

(1)收集已知矩阵信息 

通过让用户打分或者收集用户的行为数据 

(2)从已知矩阵推测未知矩阵信息 

通过基于内容的方法、协同过滤方法或者矩阵分解方法推测未知矩阵信息 

(3)评价推测方法 

常用的标准是RMSE均方根误差  
