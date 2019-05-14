# #####################################################################
# **skeletonize vs skeletonize 3d**
#
# ``skeletonize`` [Zha84]_ works by making successive passes of
# the image, removing pixels on object borders. This continues until no
# more pixels can be removed.  The image is correlated with a
# mask that assigns each pixel a number in the range [0...255]
# corresponding to each possible pattern of its 8 neighbouring
# pixels. A look up table is then used to assign the pixels a
# value of 0, 1, 2 or 3, which are selectively removed during
# the iterations.
#
# ``skeletonize_3d`` [Lee94]_ uses an octree data structure to examine a 3x3x3
# neighborhood of a pixel. The algorithm proceeds by iteratively sweeping
# over the image, and removing pixels at each iteration until the image
# stops changing. Each iteration consists of two steps: first, a list of
# candidates for removal is assembled; then pixels from this list are
# rechecked sequentially, to better preserve connectivity of the image.
#
# Note that ``skeletonize_3d`` is designed to be used mostly on 3-D images.
# However, for illustrative purposes, we apply this algorithm on a 2-D image.
#
# .. [Zha84] A fast parallel algorithm for thinning digital patterns,
#            T. Y. Zhang and C. Y. Suen, Communications of the ACM,
#            March 1984, Volume 27, Number 3.
#
# .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
#            via 3-D medial surface/axis thinning algorithms.
#            Computer Vision, Graphics, and Image Processing, 56(6):462-478,
#            1994.


import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.io import imread
from skimage.filters import threshold_otsu


data = imread('CharImage/hangul_882.jpeg')

global_thresh = threshold_otsu(data)
binary_global = data > global_thresh

data = binary_global

skeleton = skeletonize(data)
skeleton3d = skeletonize_3d(data)

fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeletonize')
ax[1].axis('off')

ax[2].imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('skeletonize_3d')
ax[2].axis('off')

fig.tight_layout()
plt.show()
