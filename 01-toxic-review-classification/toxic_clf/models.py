import datasets
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def classifier(dataset: datasets.Dataset, model):
    if model != "classic_ml":
        raise ValueError(f"Sorry, model {model} isn't implemented yet :(")

    data = dataset.to_pandas()
    X, y = data['message'], data['is_toxic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    depths = []
    scores = []

    for depth in range(2400, 2800, 25):
        tr = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        kf = KFold(n_splits=10)
        score = cross_val_score(tr, X_train, y_train, cv=kf, scoring='f1_micro').mean()
        print(f"max_depth = {depth}; mean = {score.mean()}; std = {score.std()}")
        depths.append(depth)
        scores.append(score)

    table = pd.DataFrame({ 'depth': depths, 'score': scores })
    table.plot(x='depth', y='score')

    best_depth = table['depth'][table['score'].idxmax()]

    X_test = vectorizer.transform(X_test)

    tr = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
    tr.fit(X_train, y_train)
    predictions = tr.predict(X_test)
    score = f1_score(predictions, y_test)
    print(f"f1_score", score)

    print("Confussion matrix:")
    print(confusion_matrix(y_test, predictions))
