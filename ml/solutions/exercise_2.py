import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

np.random.seed(1234)
data = read_titanic()

X = data[["Sex_Code", "Pclass_Code", "Fare", "Age"]]
y = data["Survived_Code"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# Use linear kernel
reg = SVC(kernel="linear")
reg.fit(X_train, y_train)
prediction_linear = reg.predict(X_test)

# Use the rbf kernel
reg_rbf = SVC(kernel="rbf")
reg_rbf.fit(X_train, y_train)
prediction_rbf = reg_rbf.predict(X_test)

fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(10, 10))
plots.plot_bars_and_confusion(
    truth=y_test, prediction=prediction_linear, axes=[ax1, ax2], vmin=0, vmax=182
)
plots.plot_bars_and_confusion(
    truth=y_test, prediction=prediction_rbf, axes=[ax3, ax4], vmin=0, vmax=182
)
ax1.set_title("Linear Kernel")
ax3.set_title("Radial Kernel")
ax1.set_xlim([0, 300])
ax3.set_xlim([0, 300])
None
