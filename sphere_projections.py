from time import time
import napari
import numpy as np
from sklearn.manifold import TSNE
import umap

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

xi, yi, zi = sample_spherical(10000)
coords = np.asarray(list(zip(xi, yi, zi)))

start = time()
sphere_coords_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='pca').fit_transform(coords)
duration = time() - start 
print(duration)

# start = time()
# sphere_umap = umap.UMAP().fit_transform(coords)
# duration = time() - start 
# print(duration)

viewer = napari.Viewer()
viewer.add_points(coords*100, size=1)

napari.run()
