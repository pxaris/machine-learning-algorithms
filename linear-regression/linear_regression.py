import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sgd(X, Y, learning_rate, epochs):
    [w, b] = [0.0, 0.0]

    for epoch in range(epochs):
        w_gradient = 0.0
        b_gradient = 0.0
        N = len(X)

        for i in range(N):
            w_gradient += -2*X[i] * (Y[i] - (w*X[i] + b))
            b_gradient += -2*(Y[i] - (w*X[i] + b))

        w = w - (1/float(N))*w_gradient*learning_rate
        b = b - (1/float(N))*b_gradient*learning_rate

        cost = 0.0
        for i in range(N):
            cost += (Y[i] - (w*X[i] + b))**2

        mse = np.sqrt(cost / float(N))

        print("Epoch={} , mse={}".format(epoch, mse))

    print("w={}, b={}".format(w, b))


if __name__ == "__main__":

    # hyper-parameters
    TEST_SIZE = 0.2
    LEARNING_RATE = 0.00001
    EPOCHS = 30
    random_seed = 7

    # (1) read data by specifying the proper indexes
    df_1a = pd.read_csv('data/data1a.csv', usecols=[5, 6])
    df_1b = pd.read_csv('data/data1b.csv', usecols=[0, 1])

    # (2)
    print("Correlation between x,y in 1a dataset: {}\n".format(np.corrcoef(df_1a.likes, df_1a.views)))
    print("Correlation between x,y in 1b dataset: {}\n".format(np.corrcoef(df_1b.x, df_1b.y)))

    # (3)
    plt.scatter(df_1a.likes, df_1a.views, marker='o', c='g')
    plt.title('Dataset 1a')
    plt.xlabel('likes')
    plt.ylabel('views')
    plt.savefig("1a_likes_views.png")
    plt.show()
    plt.scatter(df_1b.x, df_1b.y, marker='o', c='g')
    plt.title('Dataset 1b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("1b_x_y.png")
    plt.show()

    # (4) normalize data
    df_1a_norm = df_1a.assign(X_norm=(df_1a.likes - df_1a.likes.min()) / (df_1a.likes.max() - df_1a.likes.min()))
    df_1a_norm = df_1a_norm.assign(Y_norm=(df_1a_norm.views - df_1a_norm.views.min()) /
                                          (df_1a_norm.views.max() - df_1a_norm.views.min()))

    df_1b_norm = df_1b.assign(X_norm=(df_1b.x - df_1b.x.min()) / (df_1b.x.max() - df_1b.x.min()))
    df_1b_norm = df_1b_norm.assign(Y_norm=(df_1b_norm.y - df_1b_norm.y.min()) /
                                          (df_1b_norm.y.max() - df_1b_norm.y.min()))

    # (5) split data to train and test set
    test_df_1a = df_1a_norm.sample(frac=TEST_SIZE, random_state=random_seed)
    train_df_1a = df_1a_norm.drop(test_df_1a.index)

    test_df_1b = df_1b_norm.sample(frac=TEST_SIZE, random_state=random_seed)
    train_df_1b = df_1b_norm.drop(test_df_1b.index)

    # use specific columns for train and test set
    # normalized
    X_n_test_1a = test_df_1a.iloc[:, 2].values
    Y_n_test_1a = test_df_1a.iloc[:, 3].values
    X_n_test_1b = test_df_1b.iloc[:, 2].values
    Y_n_test_1b = test_df_1b.iloc[:, 3].values
    X_n_train_1a = train_df_1a.iloc[:, 2].values
    Y_n_train_1a = train_df_1a.iloc[:, 3].values
    X_n_train_1b = train_df_1b.iloc[:, 2].values
    Y_n_train_1b = train_df_1b.iloc[:, 3].values

    # not normalized data
    X_test_1a = test_df_1a.iloc[:, 1].values
    Y_test_1a = test_df_1a.iloc[:, 0].values
    X_test_1b = test_df_1b.iloc[:, 1].values
    Y_test_1b = test_df_1b.iloc[:, 0].values
    X_train_1a = train_df_1a.iloc[:, 1].values
    Y_train_1a = train_df_1a.iloc[:, 0].values
    X_train_1b = train_df_1b.iloc[:, 1].values
    Y_train_1b = train_df_1b.iloc[:, 0].values

    # SGD
    # not normalized data
    print("\n1a Dataset - not normalized:")
    sgd(X_train_1a, Y_train_1a, LEARNING_RATE, EPOCHS)

    print("\n1b Dataset - not normalized:")
    sgd(X_train_1b, Y_train_1b, LEARNING_RATE, EPOCHS)

    # normalized data
    print("\n1a Dataset - normalized:")
    sgd(X_n_train_1a, Y_n_train_1a, LEARNING_RATE, EPOCHS)

    print("\n1b Dataset - normalized:")
    sgd(X_n_train_1b, Y_n_train_1b, LEARNING_RATE, EPOCHS)
