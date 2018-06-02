
# coding: utf-8

# ### Today:
# * Intrudiction to Computer Vision
#     * Basic operations
#     * GUI featuers
#     * Image processing
#         * Changing color spaces
#         * Geometric Transformations
#         * Image thresholding
#         * Smoothing
#         * Morphological Transformations
#         * Contours
# * Video Analysis
#     * Background Subtraction
#     * Ball finding
# 
# ### Resources:
# * Official documentation: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
# * Good blog about CV: https://www.pyimagesearch.com/

# In[20]:


from matplotlib import pyplot as plt

import cv2
import numpy as np


# In[21]:


cv2.__version__


# In[22]:


PATH = '../../data/09_CV/'


# # Basic operations

# ## Read Image

# In[ ]:


img = cv2.imread(PATH + 'lena.png')


# ## Write Image

# In[ ]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(PATH + 'lena_processed.png', gray)


# ## Drawing Image

# In[ ]:


# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Drawing line
img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# Drawing rectangle
img = cv2.rectangle(img, (384, 10), (500, 120), (0, 255, 0), 3)

# Drawing circle
img = cv2.circle(img, (100, 100), 20, (0, 0, 255), -1, cv2.LINE_AA) # -1, 0

# Drawing text
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'OpenCV', (10, 200), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ## Accessing and Modifying pixel values

# In[ ]:


print(img[100, 100])
img[100, 100] = [0, 0, 0]

print(img.item(10, 10, 1)) # 0, 1, 2 for each channel
img.itemset((10, 10, 2), 100)


# ## Image Properties

# In[ ]:


print('Shape: ', img.shape)
print('Size: ', img.size)
print('Type: ', img.dtype)


# ## Image ROI

# In[ ]:


roi = img[200:400, 200:400]


# ## Splitting and Merging Image Channels

# In[ ]:


b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))


# ## Making Borders for Images

# In[ ]:


img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0])


# # Image Processing

# ## Changing color spaces

# In[ ]:


# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


# ## Geometric transformations

# In[ ]:


roi = img[200:300, 200:300]
# INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
res = cv2.resize(roi, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)


# ## Image thresholding

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding

# In[ ]:


from matplotlib import pyplot as plt

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()


# In[ ]:


from matplotlib import pyplot as plt


img = cv2.medianBlur(img, 5)

ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# ## Smoothing

# In[ ]:


kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)

cv2.imshow('Averaging filter', np.hstack([img, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


blur = cv2.blur(img, (10, 10))
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)

cv2.imshow('Blur filter', np.hstack([img, blur]))
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Morphological Transformations

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-ops

# In[ ]:


img = cv2.imread(PATH + 'j.png', 0)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations = 1)


# In[ ]:


img = cv2.imread(PATH + 'j.png', 0)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(img, kernel, iterations = 1)


# In[12]:


# opening = erosion followed by dilation
img = cv2.imread(PATH + 'opening.png', 0)
kernel = np.ones((5, 5), np.uint8)
res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# In[15]:


# opening = erosion followed by dilation
img = cv2.imread(PATH + 'closing.png', 0)
kernel = np.ones((5, 5), np.uint8)
res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# In[17]:


# Show image
cv2.imshow('Title', np.hstack([img, res]))
cv2.waitKey(0)
cv2.destroyAllWindows()

