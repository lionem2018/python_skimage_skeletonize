######################################################################
# **Morphological thinning**
#
# Morphological thinning, implemented in the `thin` function, works on the
# same principle as `skeletonize`: remove pixels from the borders at each
# iteration until none can be removed without altering the connectivity. The
# different rules of removal can speed up skeletonization and result in
# different final skeletons.
#
# The `thin` function also takes an optional `max_iter` keyword argument to
# limit the number of thinning iterations, and thus produce a relatively
# thicker skeleton.

from skimage.morphology import skeletonize, thin
from skimage.io import imread
from skimage.filters import threshold_otsu, threshold_local
import matplotlib.pyplot as plt

image = imread('CharImage/hangul_4.jpeg')
global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 35
binary_adaptive = threshold_local(image, block_size, offset=10)

image = binary_global

skeleton = skeletonize(image)
thinned = thin(image)
thinned_partial = thin(image, max_iter=25)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeleton')
ax[1].axis('off')

ax[2].imshow(thinned, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('thinned')
ax[2].axis('off')

ax[3].imshow(thinned_partial, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('partially thinned')
ax[3].axis('off')

fig.tight_layout()
plt.show()