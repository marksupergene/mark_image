import math
import numpy as np
import dlib
import cv2

def loadImageYcb(image):
  ycbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  ycbImage = np.float32(ycbImage)
  return ycbImage