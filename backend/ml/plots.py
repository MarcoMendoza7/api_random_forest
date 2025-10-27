import io, base64
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import numpy as np

def plot_tree_image(estimator, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    tree.plot_tree(estimator, feature_names=feature_names, filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_decision_boundary(reg, X, y, resolution=500):
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    x1, x2 = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution)
    )
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = reg.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.figure(figsize=(10, 5))
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    plt.xlabel('min_flowpktl')
    plt.ylabel('flow_fin')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
