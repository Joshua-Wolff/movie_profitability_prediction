from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class Classifier(BaseEstimator):
    def __init__(self):
        clf1 = RandomForestClassifier(random_state=0)
        clf2 = GradientBoostingClassifier(random_state=0)
        clf3 = AdaBoostClassifier(random_state=0)
        self.clf = VotingClassifier(estimators=[('rfc', clf1), ('gbc', clf2), ('adac', clf3)], voting='soft')

    def fit(self, X, y):

        params = {'rfc__max_depth': [5, 6, 7], \
            'gbc__n_estimators': [10, 100], 'gbc__max_depth' : [2, 3], \
            'adac__n_estimators' : [10, 100]}

        grid = GridSearchCV(estimator=self.clf, scoring='accuracy', param_grid=params, cv=5)
        grid = grid.fit(X, y)

        self.clf = grid.best_estimator_
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)