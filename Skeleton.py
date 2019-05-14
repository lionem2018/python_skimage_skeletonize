"""
===========
Skeletonize
===========

Skeletonization reduces binary objects to 1 pixel wide representations. This
can be useful for feature extraction, and/or representing an object's topology.

``skeletonize`` works by making successive passes of the image. On each pass,
border pixels are identified and removed on the condition that they do not
break the connectivity of the corresponding object.
"""
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.io import imread
from skimage.filters import threshold_otsu


# Invert the horse image
# image = invert(data.horse())
image = imread('CharImage/Char_31092.jpeg')

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

image = binary_global

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()