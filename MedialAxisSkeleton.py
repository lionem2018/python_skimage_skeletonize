######################################################################
# **Medial axis skeletonization**
#
# The medial axis of an object is the set of all points having more than one
# closest point on the object's boundary. It is often called the *topological
# skeleton*, because it is a 1-pixel wide skeleton of the object, with the same
# connectivity as the original object.
#
# Here, we use the medial axis transform to compute the width of the foreground
# objects. As the function ``medial_axis`` returns the distance transform in
# addition to the medial axis (with the keyword argument ``return_distance=True``),
# it is possible to compute the distance to the background for all points of
# the medial axis with this function. This gives an estimate of the local width
# of the objects.
#
# For a skeleton with fewer branches, ``skeletonize`` or ``skeletonize_3d``
# should be preferred.

from skimage.morphology import medial_axis, skeletonize, skeletonize_3d
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Generate the data
data = imread('CharImage/hangul_32.jpeg')

global_thresh = threshold_otsu(data)
binary_global = data > global_thresh

data = binary_global

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)

# Compare with other skeletonization algorithms
skeleton = skeletonize(data)
skeleton3d = skeletonize_3d(data)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(dist_on_skel, cmap='magma', interpolation='nearest')
ax[1].contour(data, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')

ax[2].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('skeletonize')
ax[2].axis('off')

ax[3].imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('skeletonize_3d')
ax[3].axis('off')

fig.tight_layout()
plt.show()
