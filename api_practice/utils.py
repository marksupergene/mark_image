import cv2
import numpy as np
import ssl
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json
import os
from six.moves import urllib

def parseRequest(request): 
  return json.loads(request.body.decode('utf-8'))

def url_to_image(url, alpha = 1):
  image = cv2.imread(url)
  if image is not None:
    return image
  else:
    resp = urllib.request.urlopen(url, context=ssl.SSLContext())
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, alpha)
    # else:
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def image_to_url(path, image):
  cv2.imwrite(path, image)
  a = cloudinary.uploader.upload(path)
  os.remove(path)
  return a['url']
  # return ""
 
def video_to_url(path):
  a = cloudinary.uploader.upload(path, resource_type = "video")
  os.remove(path)
  return a['url']