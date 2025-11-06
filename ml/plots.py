import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import collections
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex


rng = np.random.default_rng(0)


colors = ["xkcd:sky", "xkcd:grass"]
cmap = ListedColormap(colors)


def create_discrete_colormap(n_classes):
    if n_classes == 2:
        return cmap.copy()
    return ListedColormap([f"C{i}" for i in range(n_classes)])


def set_plot_style():
    sns.reset_orig()
    plt.rcParams["figure.figsize"] = (9.23, 9.23 / 3 * 2)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.max_open_warning"] = 50
    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def twospirals(n_samples, noise=0.5, rng=rng):
    """
    Returns the two spirals dataset.
    """
    n = np.sqrt(rng.uniform(size=(n_samples, 1))) * 360 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + rng.uniform(size=(n_samples, 1)) * noise
    d1y = np.sin(n) * n + rng.uniform(size=(n_samples, 1)) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_samples), np.ones(n_samples))),
    )


def draw_linear_regression_function(reg, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()

    if reg.coef_.ndim > 1:
        b_1, b_2 = reg.coef_[0, :]
    else:
        b_1, b_2 = reg.coef_

    b_0 = reg.intercept_

    # solve the function y = b_0 + b_1*X_1 + b_2 * X_2 for X2
    x_low, x_high = ax.get_xlim()
    x1s = np.linspace(x_low, x_high)
    x2s = (0.5 - b_0 - b_1 * x1s) / b_2

    ax.plot(x1s, x2s, **kwargs)


def plot_3d_views(X, y, cmap=cmap):
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig, axs = plt.subplots(
        2, 2, subplot_kw={"projection": "3d"}, constrained_layout=False
    )

    for ax in axs.ravel():
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap, lw=0)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    axs[0, 1].view_init(0, 0)
    axs[1, 0].view_init(0, 90)
    axs[1, 1].view_init(90, 0)
    fig.subplots_adjust(wspace=0.005, hspace=0.005)


def draw_tree(clf):
    import pydotplus

    d = tree.export_graphviz(clf, out_file=None, filled=True)
    graph = pydotplus.graph_from_dot_data(d)

    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(to_hex(colors[i]))

    return graph.create(format="png")


def draw_svm_decision_function(clf, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()

    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    x1 = np.linspace(x_low, x_high, 40)
    x2 = np.linspace(y_low, y_high, 40)

    X1, X2 = np.meshgrid(x1, x2)
    xy = np.vstack([X1.ravel(), X2.ravel()]).T
    # get the separating hyperplane
    Z = clf.decision_function(xy).reshape(X1.shape)

    # plot decision boundary and margins
    label = kwargs.pop("label", "Decision Boundary")
    cs = ax.contour(
        X1, X2, Z, levels=[-1.0, 0, 1.0], linestyles=["--", "-", "--"], **kwargs
    )
    cs.collections[0].set_label(label)
    plt.axis("off")


def draw_decision_boundaries(knn, ax=None, cmap="winter", alpha=0.07, **kwargs):
    if not ax:
        ax = plt.gca()

    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    x1 = np.linspace(x_low, x_high, 100)
    x2 = np.linspace(y_low, y_high, 100)

    X1, X2 = np.meshgrid(x1, x2)
    xy = np.vstack([X1.ravel(), X2.ravel()]).T
    Z = knn.predict(xy).reshape(X1.shape)

    label = kwargs.pop("label", "Decision Boundary")
    # plot decision boundary and margins
    cs = ax.contourf(X1, X2, Z, **kwargs, cmap=cmap, alpha=alpha)
    cs.collections[0].set_label(label)
    plt.axis("off")


def draw_decision_surface(clf, predictions, label=None):
    ax = plt.gca()
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    x1 = np.linspace(x_low, x_high, 100)
    x2 = np.linspace(y_low, y_high, 100)

    X1, X2 = np.meshgrid(x1, x2)
    xy = np.vstack([X1.ravel(), X2.ravel()]).T
    Z = clf.predict_proba(xy)[:, 1].reshape(X1.shape)

    plt.imshow(
        Z,
        extent=[x_low, x_high, y_low, y_high],
        cmap="GnBu",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    plt.grid()
    plt.colorbar(label=label)
    plt.axis("off")


def plot_bars_and_confusion(
    truth,
    prediction,
    axes=None,
    vmin=None,
    vmax=None,
    cmap="inferno",
    title=None,
    bar_color=None,
):
    accuracy = accuracy_score(truth, prediction)
    cm = confusion_matrix(truth, prediction)

    if not isinstance(truth, pd.Series):
        truth = pd.Series(truth)

    if not isinstance(prediction, pd.Series):
        prediction = pd.Series(prediction)

    correct = pd.Series(np.where(truth.values == prediction.values, "Correct", "Wrong"))

    truth.sort_index(inplace=True)
    prediction.sort_index(inplace=True)

    if not axes:
        fig, axes = plt.subplots(1, 2)

    if not vmin:
        vmin = cm.min()

    if not vmax:
        vmax = cm.max()

    if not bar_color:
        correct.value_counts().plot.barh(ax=axes[0])
    else:
        correct.value_counts().plot.barh(ax=axes[0], color=bar_color)

    axes[0].text(150, 0.5, "Accuracy {:0.3f}".format(accuracy))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")
    if title:
        plt.suptitle(title)
