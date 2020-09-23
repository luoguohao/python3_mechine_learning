from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tpot import TPOTRegressor


def classification():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        train_size=0.75, test_size=0.25, random_state=42)

    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)

    print(tpot.score(X_test, y_test))
    tpot.export('tpot_digits_pipeline.py')


def regression():
    housing = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                        train_size=0.75, test_size=0.25, random_state=42)
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_boston_pipeline.py')


if __name__ == '__main__':
    # classification()
    regression()
