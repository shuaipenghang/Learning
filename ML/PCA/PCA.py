import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

def data_init():
    df = pd.read_csv('iris.data')
    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    label_dict = {1: 'Iris-Setosa',
                  2: 'Iris-Versicolor',
                  3: 'Iris-Virginica'}

    feature_dict = {0: 'sepal length [cm]',
                    1: 'sepal width [cm]',
                    2: 'petal length [cm]',
                    3: 'sepal width [cm]'}

    plt.figure(figsize = (8, 6))
    for cnt in range(4):
        plt.subplot(2, 2, cnt + 1)
        for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
            plt.hist(X[y == lab, cnt],
                     label = lab,
                     bins = 10,
                     alpha = 0.3)
        plt.xlabel(feature_dict[cnt])
        plt.legend(loc = 'upper right', fancybox = True, fontsize = 8)
    plt.tight_layout()  # 自动填充整个领域

    X_std = StandardScaler().fit_transform(X) #标准化
    mean_vec = np.mean(X_std, axis = 0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1) #协方差矩阵
    print('Covariance matrix \n%s' %cov_mat)
#np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat) #特征值，特征向量
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))] #特征值+特征向量
    eig_pairs.sort(key=lambda x:x[0], reverse = True) #根据特征值排序

    tot = sum(eig_vals)
    var_exp = [(i / tot) *100 for i in sorted(eig_vals, reverse=True)] #方差百分比
    print(var_exp)
    cum_var_exp = np.cumsum(var_exp) #累加和

    plt.figure(figsize = (6,4))
    plt.bar(range(4), var_exp, alpha=0.5, align = 'center', label = 'individual explained variance')
    plt.step(range(4), cum_var_exp, where = 'mid', label = 'cumulative explainde variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

    #前两个特征向量
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                          eig_pairs[1][1].reshape(4, 1)))

    print('Matrix W:\n', matrix_w)

    Y = X_std.dot(matrix_w)

    #原始数据
    plt.figure(figsize = (6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
        plt.scatter(X[y==lab, 0],
                    X[y==lab, 1],
                    label = lab,
                    c = col)
    plt.ylabel('sepal_len')
    plt.xlabel('sepal_wid')
    plt.legend(loc='best')
    plt.tight_layout()


    plt.figure(figsize = (6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label = lab,
                    c = col)
    plt.ylabel('Principal Componet 2')
    plt.xlabel('Principal Componet 1')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_init()