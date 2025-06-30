from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(1235)

data = read_titanic()

X = data[["Sex_Code", "Pclass_Code", "Fare", "Age"]]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

df = pd.DataFrame()
ps = ParameterGrid({"max_depth": range(1, 20), "criterion": ["entropy", "gini"]})
for d in ps:
    clf = DecisionTreeClassifier(max_depth=d["max_depth"], criterion=d["criterion"])
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    df = df.append(
        {"max_depth": d["max_depth"], "criterion": d["criterion"], "accuracy": acc},
        ignore_index=True,
    )

df = df.pivot("max_depth", "criterion", "accuracy")
sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".3f")
None
