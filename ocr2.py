import numpy as np
from PIL import Image


threshold = 100

img = Image.open('temp/uploaded_image.png')
# Note, if you load a color image, also apply img.convert('L')

img_np = np.array(img)
img_np = np.where(img_np > threshold, 255, 0)

img.putdata(img_np.flatten())

img.save('img_thresholded.png')
