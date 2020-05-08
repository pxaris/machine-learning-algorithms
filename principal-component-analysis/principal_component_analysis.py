from numpy import genfromtxt
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression


def calc_pca_from_scratch(input_file, output_file="data/foo.csv", n_components=3):
    with open('pca_on_demo3b.txt', 'w+') as file:
        # load dataset
        data = genfromtxt(input_file, delimiter=',')

        # calculate mean for each feature and normalize data
        M = mean(data.T, axis=1)
        data_normal = data - M

        # calculate covariance matrix
        covariance = cov(data_normal.T)
        file.write("\nThe covariance matrix of the normalized data is the following: \n")
        file.write(str(covariance))

        # calculate eigenvalues and eigenvectors of covariance matrix
        values, vectors = eig(covariance)
        file.write("\n\nThe eigenvalues of the normalized data are the following: \n")
        file.write(str(values))

        # select the most important eigenvalues and eigenvectors
        new_values = values[0:n_components]
        file.write("\n\nThe most important eigenvalues are the following: \n")
        file.write(str(new_values))
        new_vectors = vectors[0:n_components]
        file.write("\n\nThe most important eigenvectors are the following: \n")
        file.write(str(new_vectors))

        # apply the most important eigenvectors to the dataset
        # in order to get a new one reduced in dimensions
        new_data = new_vectors.dot(data_normal.T)

        # save the reduced dataset to a CSV file
        np.savetxt(output_file, new_data.T, delimiter=",")
        file.write('\n\nReduced dataset saved at: {}\n'.format(output_file))


def apply_logistic_regression(x, y, title='', results_file='output.txt'):
    with open(results_file, 'a+') as file:
        # Split the dataset into train and test set
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2)

        # fit Logistic Regression
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(Xtrain, Ytrain)

        # Get the predictions on the test set
        prediction = classifier.predict(Xtest)

        # Calculate the total number of errors on the test set
        errors = 0
        for index in range(0, len(prediction) - 1):
            if prediction[index] != Ytest[index]:
                errors += 1

        file.write("\n{}\n".format(title))
        file.write("Total errors on the test dataset: {}\n".format(errors))


if __name__ == '__main__':
    # (1) Apply PCA to demo3b.csv dataset
    input_file = 'data/demo3b.csv'
    calc_pca_from_scratch(input_file)
    calc_pca_from_scratch(input_file, output_file="data/foo2.csv", n_components=2)
    calc_pca_from_scratch(input_file, output_file="data/foo1.csv", n_components=1)

    # (2) Compare Logistic Regression Performance
    # on initial and reduced with PCA data

    # begin with initial data
    df = pd.read_csv('data/demo3a.csv')

    # Separate the input features from the target variable
    x = df.iloc[:, 1:13].values
    y = df.iloc[:, 0].values

    apply_logistic_regression(x, y, title='Logistic regression on initial data')

    # continue with data after PCA - 3 components
    df = pd.read_csv('data/foo.csv')
    x = df.iloc[:, :].values
    apply_logistic_regression(x, y, title='Logistic regression on data after PCA-3 components')

    # continue with data after PCA - 2 components
    df = pd.read_csv('data/foo2.csv')
    x = df.iloc[:, :].values
    apply_logistic_regression(x, y, title='Logistic regression on data after PCA-2 components')

    # continue with data after PCA - 1 component
    df = pd.read_csv('data/foo1.csv')
    x = df.iloc[:, :].values
    apply_logistic_regression(x, y, title='Logistic regression on data after PCA-1 component')
