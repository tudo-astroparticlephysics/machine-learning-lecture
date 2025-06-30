from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm.auto import tqdm
import numpy as np


k = 4
n_iters = 25
discrete_cmap = ListedColormap([f"C{i}" for i in range(k)])
fps = 25
interval = 1000 / fps
time_per_iter = 1
frames = n_iters * time_per_iter * fps

# choose inital cluster centers
X, y = make_blobs(
    n_samples=500,
    centers=k,
    center_box=(-2, 2),
    cluster_std=0.5,
    random_state=1,
)

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect(1)
ax.set_axis_off()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)


init_centers = np.random.uniform(-1, 1, size=[k, 2])
center_history = np.zeros((n_iters, k, 2))


center_lines = [ax.plot([], [])[0] for _ in range(k)]
points = ax.scatter(X[:, 0], X[:, 1], c="k", cmap=discrete_cmap, alpha=0.5)
center_plot = ax.scatter(
    init_centers[:, 0],
    init_centers[:, 1],
    c=np.arange(k),
    cmap=discrete_cmap,
    marker="h",
    edgecolor="k",
    s=400,
    label="cluster center",
)

ax.legend(loc="upper right")


def init():
    t = ax.set_title("iteration  0")
    return *center_lines, points, t


def update(frame, bar=None):
    if bar is not None:
        bar.update(1)

    i = frame // (fps * time_per_iter)
    if i > 0:
        kmeans = KMeans(n_clusters=k, init=init_centers, max_iter=i + 1, n_init=1)
        prediction = kmeans.fit_predict(X)
        center_history[i] = kmeans.cluster_centers_
        center_plot.set_offsets(kmeans.cluster_centers_)
        points.set_array(prediction)
    else:
        center_history[i] = init_centers

    for j, line in enumerate(center_lines):
        line.set_data(center_history[: i + 1, j, 0], center_history[: i + 1, j, 1])

    points.set_cmap(discrete_cmap)
    t = ax.set_title("iteration {}".format(i + 1))

    return *center_lines, points, t


bar = tqdm(total=frames)
ani = FuncAnimation(
    fig,
    update,
    blit=True,
    init_func=init,
    frames=frames,
    fargs=(bar,),
    interval=interval,
)
ani.save("kmeans_clustering.mp4")
ani.pause()
plt.close(fig)
