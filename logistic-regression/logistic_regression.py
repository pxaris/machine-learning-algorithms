import pandas as pd
import sklearn.metrics as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # (1) load csv file
    # data acquired from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
    # format: Each line is composed by two columns: one with label (ham or spam) and other with the raw text
    df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None)
    print(df.describe())

    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values

    # (2) split dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # vectorize data
    vectorizer = TfidfVectorizer('english')
    vectorizer.fit(X_train)
    Xtrain = vectorizer.transform(X_train)
    Xtest = vectorizer.transform(X_test)

    # (3) fit Logistic Regression model
    model = LogisticRegression()
    model.fit(Xtrain, y_train)

    # classify patterns of test set
    y_pred = model.predict(Xtest)

    # (4) calculate accuracy based on test set
    with open('logistic_regression_performance.txt', 'w+') as file:
        accuracy = sm.accuracy_score(y_test, y_pred)
        file.write('\nAccuracy of Logistic Regression is: {}%\n'.format(round(accuracy*100, 2)))
