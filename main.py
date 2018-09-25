def main():
    import scipy
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    iris = datasets.load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=15
    )

    classifier = GaussianNB()
    model = classifier.fit(x_train, y_train)

    targets_predicted = model.predict(x_test)

    numCorrect = 0
    numTotal = 0

    for i in range(len(targets_predicted)):
        if targets_predicted[i] == y_test[i]:
            numCorrect += 1

        numTotal += 1


    accuracy = (numCorrect/numTotal) * 100

    print("Gaussian was %.2f %% accurate" % (accuracy))


if __name__ == '__main__':
    main()
