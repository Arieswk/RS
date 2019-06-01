# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error,confusion_matrix,precision_recall_curve
from math import sqrt
import matplotlib.pyplot as plt
import itertools
from scipy.sparse.linalg import svds
from operator import itemgetter

# 评分数据协同过滤预测
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# RMSE评测预测矩阵
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

# precision、recall、f-1值、train为训练好的数据
def Precision_Recall_F1Score(train,test,N):
    hit = 0
    n_recall = 0
    n_precision = 0
    for user in range(n_users):
        tu = list(test[user].nonzero()[0] + 1)
        rank_dict = {}
        for i in range(len(train[user])):
            rank_dict[i + 1] = train[user][i]
        rank = sorted(rank_dict.items(), key=lambda d: d[1])
        rank = [x[0] for x in rank[:N]]
        for item in rank:
            if item in tu:
                hit += 1
        n_recall += len(tu)
        n_precision += N
    recall = hit / (1.0 * n_recall)
    precision = hit / (1.0 * n_precision)
    F1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, F1

#绘PR图：精确率-召回率曲线
def plot_precision_recall():
    plt.step(recall,precision,color = 'b',alpha = 0.2,where = 'post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color = 'b')
    plt.plot(recall,precision,linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精准率')
    plt.title('精确率-召回率曲线')
    plt.savefig('PR曲线.png')
    plt.show()

#加载数据
header = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep = '\t', names = header)

n_users = data.user_id.unique().shape[0]
n_items = data.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

# 随机划分数据集，test大小为0.2
train_data, test_data = train_test_split(data, test_size=0.2)

train_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1,line[2] - 1] = line[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]


#计算用户、物品cosine相似度
user_similarity = pairwise_distances(train_data_matrix,metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T,metric='cosine')

#基于物品、用户预测（协同过滤）
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#RMSE评测
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

#计算MovieLens数据集的稀疏度
sparsity = round(1.0-len(data)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

# get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix,k=20)
s_diag_matrix = np.diag(s)
# print('调参Sigma')
# print(sum(s[-300:] ** 2) / sum(s ** 2))
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

print('SVD RMSE: ' + str(rmse(X_pred, test_data_matrix)))

precision,recall,f1_score = Precision_Recall_F1Score(X_pred,test_data_matrix,1)
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('f1_score: ' + str(f1_score))

# test_y
test_y = test_data_matrix
test_y[test_y > 0] = 1
test_y = test_y.flatten()


# 计算精确率，召回率，阈值用于可视化
precision,recall,threshold = precision_recall_curve(test_y,X_pred.flatten())

# 设置plt中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plot_precision_recall()
