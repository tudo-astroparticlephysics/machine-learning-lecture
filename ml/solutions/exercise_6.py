from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from matplotlib import patches

X, y = make_moons(n_samples=5000, noise=0.9)

clf = DecisionTreeClassifier(min_samples_leaf=50)
cv = StratifiedKFold(n_splits=5)

fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(12, 12))

roc_auc = []
pr_auc = []

for train, test in cv.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    clf.fit(X_train, y_train)

    prediction = clf.predict_proba(X_test)[:, 1]

    p, r, thresholds_pr = precision_recall_curve(y_test, prediction)
    fpr, tpr, thresholds_roc = roc_curve(y_test, prediction)

    roc_auc.append(roc_auc_score(y_test, prediction))
    pr_auc.append(average_precision_score(y_test, prediction))

    ax1.step(thresholds_pr, r[:-1], color="gray", where="post")
    ax1.step(thresholds_pr, p[:-1], color="darkgray", where="post")

    ax2.step(r, p, color="darkmagenta", where="post")

    ax3.step(thresholds_roc, tpr, color="gray", where="post")
    ax3.step(thresholds_roc, fpr, color="darkgray", where="post")

    ax4.step(fpr, tpr, color="mediumvioletred", where="post")


p1 = patches.Patch(color="gray", label="Recall")
p2 = patches.Patch(color="darkgray", label="Precission")
ax1.legend(handles=[p1, p2])
ax1.set_xlabel("Decission Threshold")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_ylabel("Precission")
ax2.set_xlabel("Recall")
s = "AUC {:0.3f} +/- {:0.3f}".format(np.array(pr_auc).mean(), np.array(pr_auc).std())
ax2.text(0.2, 0.2, s)

p1 = patches.Patch(color="gray", label="True Positive Rate")
p2 = patches.Patch(color="darkgray", label="False Positive Rate")
ax3.legend(handles=[p1, p2])
ax3.set_xlabel("Decission Threshold")
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.set_ylabel("True Positive Rate")
ax4.set_xlabel("False Positive Rate")
s = "AUC {:0.3f} +/- {:0.3f}".format(np.array(roc_auc).mean(), np.array(roc_auc).std())
ax4.text(0.2, 0.2, s)

None
