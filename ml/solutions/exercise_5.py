from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

np.random.seed(1234)
data = read_titanic()

X = data[["Sex_Code", "Pclass_Code", "Fare", "Age"]]
y = data["Survived_Code"]

svc = SVC(kernel="linear")
knn = KNeighborsClassifier(n_neighbors=5)
tree = DecisionTreeClassifier(max_depth=5)

results = []
for clf, name in zip([svc, knn, tree], ["SVM", "kNN", "tree"]):
    r = cross_validate(
        clf, X=X, y=y, cv=5, scoring=["accuracy", "precision", "recall", "f1"]
    )
    df = pd.DataFrame().from_dict(r)
    df["classifier"] = name
    results.append(df)

df = pd.concat(results).drop(["fit_time", "score_time"], axis="columns")

means = df.groupby("classifier").mean()
deviations = df.groupby("classifier").std()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
sns.heatmap(means, cmap="viridis", annot=True, ax=ax1, vmin=0, vmax=1)
sns.heatmap(deviations, cmap="viridis", annot=True, ax=ax2, vmin=0, vmax=1)
