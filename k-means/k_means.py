import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np


def generate_fig_elbow_method(data, filename='kmeans_elbow_method.png'):
    # calculate distortion for a range of number of cluster
    distortions = []
    for k in range(1, 15):
        k_means = KMeans(
            n_clusters=k, init='random',
            n_init=10, max_iter=1000,
            tol=1e-04, random_state=0
        )
        k_means.fit(data)
        distortions.append(k_means.inertia_)

    # plot and save figure
    plt.figure(figsize=(12, 12))
    plt.plot(range(1, 15), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig(filename)
    print('Figure saved at: {}'.format(filename))


def plot_data_for_all_feature_tuples(data, labels, title='', filename='fig.png'):
    colors = np.array(['r', 'g', 'b'])
    plt.figure(figsize=(15, 10))
    idx = 1
    for i, column in enumerate(data.columns):
        for j in range(i + 1, len(data.columns)):
            plt.subplot(2, 3, idx)
            secondary_col = data.columns[j]
            plt.scatter(data[column], data[secondary_col], c=colors[labels])
            plt.xlabel(column)
            plt.ylabel(secondary_col)
            idx += 1

    plt.suptitle(title, fontsize=14)
    plt.savefig(filename)
    print('Figure saved at: {}'.format(filename))


if __name__ == '__main__':
    # load iris data
    iris = datasets.load_iris()

    # Store the inputs as a Pandas Dataframe and set the column names
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']

    # (1) Elbow method
    generate_fig_elbow_method(x)
    # the 'knee' of the curve happens on k=3

    # (2) apply k-means for the optimal #clusters
    k_means = KMeans(
            n_clusters=3, init='random',
            n_init=10, max_iter=1000,
            tol=1e-04, random_state=0
        )
    k_means.fit(x)
    print('\nK-means labels:\n{}'.format(k_means.labels_))
    # how k-means works?

    # (3) Plot real classes as well as k-means clusters
    plot_data_for_all_feature_tuples(x, y['Targets'], title='Real classes', filename='kmeans_real_classes.png')
    plot_data_for_all_feature_tuples(x, k_means.labels_, title='K-means clusters', filename='kmeans_clusters.png')

    # (4) Measure performance of K-means
    with open('kmeans_performance.txt', 'w+') as file:
        accuracy = sm.accuracy_score(y, k_means.labels_)
        file.write('\nAccuracy of K-means is: {}%\n'.format(round(accuracy*100, 2)))

        error_matrix = sm.confusion_matrix(y, k_means.labels_)
        file.write('\nK-means confusion matrix:\n{}'.format(error_matrix))
