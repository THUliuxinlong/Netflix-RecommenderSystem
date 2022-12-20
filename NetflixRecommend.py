import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    # 导入数据
    user = pd.read_csv("./data/users.txt", names = ['userid'])
    netflix_train = pd.read_csv("./data/netflix_train.txt", sep=' ', names = ['user_id', 'film_id', 'rating', 'date'])
    netflix_test = pd.read_csv("./data/netflix_test.txt", sep=' ', names = ['user_id', 'film_id', 'rating', 'date'])

    # 加一列，给用户从零开始进行编号,并添加到训练集和测试集上
    user['id'] = range(len(user))
    netflix_train = netflix_train.merge(user, left_on='user_id', right_on='userid')
    netflix_test = netflix_test.merge(user, left_on='user_id', right_on='userid')

    # 通过数据透视函数构建用户*电影矩阵
    X_train = netflix_train.pivot(index='id', columns='film_id', values='rating')
    X_test = netflix_test.pivot(index='id', columns='film_id', values='rating')

    # 测试集缺失部分电影，先用nan补齐为10000*10000矩阵，后续根据计算需要再用0填充
    for i in range(1, 10001):
        if i not in X_test.columns:
            X_test[i] = np.nan
    X_test = X_test.sort_index(axis=1)

    X_train = X_train.fillna(0)

    return X_train, X_test, netflix_test


def Collaborate_Filtering(X_train, X_test, netflix_test):
    # 计算cos相似度矩阵,返回数组的第i行第j列表示a[i]与a[j]的余弦相似度
    similarity_matrix = cosine_similarity(X_train)
    # 基于协同过滤计算作预测
    X_train = np.array(X_train)
    for i in range(X_train.shape[0]):
        # 用argsort()函数求出与用户i最相似的用户，按照相似度倒序排列成列表indexs
        indexs = np.argsort(similarity_matrix[i, :])[::-1]
        for j in range(X_train.shape[1]):
            if X_train[i, j] == 0:
                sum = 0
                num = 0
                simi = 0
                k = 0
                # 找出看过此电影的相似度排名前5的用户并计算加权平均值
                while num < 5 & k < X_train.shape[1]:
                    if X_train[indexs[k], j] > 0:
                        sum = sum + similarity_matrix[i, indexs[k]] * X_train[indexs[k], j]
                        simi = simi + similarity_matrix[i, indexs[k]]
                        k = k+1
                        num = num + 1
                    else:
                        k = k+1
                if simi != 0:
                    X_train[i, j] = sum/simi
                else:
                    X_train[i, j] = 0
            else:
                continue
    # RMSE
    RMSE = np.sqrt(np.sum(np.sum(np.square(X_train - X_test)))/netflix_test.shape[0])
    print('RMSE', RMSE)
    return RMSE


def Matrix_Decomposition(X_train, X_test, netflix_test, alpha, K, lamda, epochs):
    A = X_train > 0
    X_train = np.array(X_train.fillna(0))
    U = np.random.randn(10000, K)*0.1
    V = np.random.randn(10000, K)*0.1
    alpha = alpha
    lamda = lamda
    # 梯度下降
    J = np.zeros((epochs))
    RMSE = np.zeros((epochs))
    for i in range(epochs):
        dU = np.dot(np.multiply(A, (np.dot(U, V.T) - X_train)), V) + 2 * lamda * U
        dV = np.dot(np.multiply(A, (np.dot(U, V.T) - X_train)), U) + 2 * lamda * V
        old_U = U
        old_V = V
        U = U - alpha / (1 + 0.1 * i) * dU  # Learning rate decay
        V = V - alpha / (1 + 0.1 * i) * dV
        J[i] = 1 / 2 * np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U, V.T)))))) + lamda * np.sum(
            np.sum(np.square(U))) \
                  + lamda * np.sum(np.sum(np.square(V)))
        RMSE[i] = np.sqrt(np.sum(np.sum(np.square(np.dot(U, V.T) - X_test))) / netflix_test.shape[0])
        print('epoch', i, 'RMSE', RMSE[i])
    return RMSE


# netflix_test用于RMSE公式中计算训练样本数
X_train, X_test, netflix_test = load_data()

# Collaborate Filtering
Collaborate_Filtering(X_train, X_test, netflix_test)

# Matrix Decomposition
epochs = 150
alpha = 0.0001
RMSE1 = Matrix_Decomposition(X_train, X_test, netflix_test, alpha=alpha, K=50, lamda=0.01, epochs=epochs)
RMSE2 = Matrix_Decomposition(X_train, X_test, netflix_test, alpha=alpha, K=50, lamda=0.001, epochs=epochs)
RMSE3 = Matrix_Decomposition(X_train, X_test, netflix_test, alpha=alpha, K=20, lamda=0.01, epochs=epochs)
RMSE4 = Matrix_Decomposition(X_train, X_test, netflix_test, alpha=alpha, K=20, lamda=0.001, epochs=epochs)

# 可视化
plt.plot(range(epochs), RMSE1, label='K=50, lamda=0.01')
plt.plot(range(epochs), RMSE2, label='K=50, lamda=0.001')
plt.plot(range(epochs), RMSE3, label='K=20, lamda=0.01')
plt.plot(range(epochs), RMSE4, label='K=20, lamda=0.001')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()
print('K=50, lamda=0.01, RMSE:', RMSE1[epochs - 1])
print('K=50, lamda=0.001, RMSE:', RMSE2[epochs - 1])
print('K=20, lamda=0.01, RMSE:', RMSE3[epochs - 1])
print('K=20, lamda=0.001, RMSE:', RMSE4[epochs - 1])
