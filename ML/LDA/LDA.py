import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from matplotlib import pyplot as plt

def data_init():
    # 读取数据
    df = pd.io.parsers.read_csv(
        filepath_or_buffer='iris.data',
        header=None,
        sep=',',
    )
    # 数据标记，利于后面使用数据
    feature_dict = {i: label for i, label in zip(
        range(4), ('sepal length in cm',
                   'sepal width in cm',
                   'petal length in cm',
                   'petal width in cm',))}

    df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
    df.dropna(how='all', inplace=True)  # 滤除缺失数据

    return df

#输入df数据
def LDA(df):
    X = df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
    y = df['class label'].values #3个类别

    #标记
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1

    #label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}
    np.set_printoptions(precision=4) #小数精度显示为4位
    mean_vectors = []
    for cl in range(1, 4):
        mean_vectors.append(np.mean(X[y==cl], axis=0))
        print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl - 1])) #三个类别的四个参数均值

    S_W = np.zeros((4, 4))
    for cl, mv in zip(range(1, 4), mean_vectors):
        class_sc_mat = np.zeros((4, 4))
        for row in X[y == cl]:
            row, mv = row.reshape(4, 1), mv.reshape(4, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat
    print('within-class Scatter Matrix:\n', S_W) #类内散度

    overall_mean = np.mean(X, axis = 0)
    S_B = np.zeros((4, 4))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i+1, :].shape[0]
        mean_vec = mean_vec.reshape(4, 1)
        overall_mean = overall_mean.reshape(4, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    print('between-class Scatter Matrix:\n', S_B) #类间散度

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) #求解出特征值与特征向量

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:, i].reshape(4, 1)
        print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))

    #特征向量表示映射方向，特征值表示特征向量的重要程度
    #make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key = lambda k: k[0], reverse = True) #排序，从高到低

    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])

    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i, j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

    W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1))) #选择前两维的特征向量，列拼接
    print('Matrix W:\n', W.real)

    X_lda = X.dot(W) #降维后的数据
    assert X_lda.shape == (150, 2), "The matrix is not 150x2 dimensional."

    return X_lda

def plot_step_lda(df, X_lda):

    label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}
    y = df['class label'].values  # 3个类别
    # 标记
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1
    ax = plt.subplot(111)
    for label, marker, color in zip(range(1, 4), ('.', 's', 'o'), ('blue', 'red', 'green')):

        plt.scatter(x = X_lda[:, 0].real[y == label],
                    y = X_lda[:, 1].real[y == label],
                    marker = marker,
                    color = color,
                    alpha = 0.5,
                    label = label_dict[label])
    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc = 'upper right', fancybox = True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    #hide axis ticks
    plt.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off',
                    labelbottom = 'on', left = 'off', right = 'off', labelleft = 'on')

    #remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = data_init()
    X_lda = LDA(df)
    plot_step_lda(df, X_lda)