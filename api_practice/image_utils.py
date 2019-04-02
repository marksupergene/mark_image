import cv2
import numpy as np

def loadImageYcb(image):
    ycbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbImage = np.float32(ycbImage)
    return ycbImage


def colorDodge(top, bottom):

  # divid the bottom by inverted top image and scale back to 250
  output = cv2.divide(bottom, 255 - top, scale=256)

  return output

def sketchPencilUsingBlending(original, kernelSize=21):
  img = np.copy(original)

  # Convert to grayscale
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Invert the grayscale image
  imgGrayInv = 255 - imgGray

  # Apply GaussianBlur
  imgGrayInvBlur = cv2.GaussianBlur(imgGrayInv, (kernelSize, kernelSize), 0)

  # blend using color dodge
  output = colorDodge(imgGrayInvBlur, imgGray)

  return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def color_transfer(src, dst):

  output = np.copy(dst)
  srcLab = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
  dstLab = np.float32(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
  outputLab = np.float32(cv2.cvtColor(output, cv2.COLOR_BGR2LAB))

  print(src)

  print(srcLab)

  # Split the Lab images into their channels
  srcL, srcA, srcB = cv2.split(srcLab)
  dstL, dstA, dstB = cv2.split(dstLab)
  outL, outA, outB = cv2.split(outputLab)
  

  outL = dstL - dstL.mean()
  outA = dstA - dstA.mean()
  outB = dstB - dstB.mean()

  # scale the standard deviation of the destination image
  outL *= srcL.std() / (dstL.std() if dstL.std() else 1)
  outA *= srcA.std() / (dstA.std() if dstA.std() else 1)
  outB *= srcB.std() / (dstB.std() if dstB.std() else 1)

  # Add the mean of the source image to get the color
  outL = outL + srcL.mean()
  outA = outA + srcA.mean()
  outB = outB + srcB.mean()

  # Ensure that the image is in the range
  # as all operations have been done using float
  outL = np.clip(outL, 0, 255)
  outA = np.clip(outA, 0, 255)
  outB = np.clip(outB, 0, 255)

  # Get back the output image
  outputLab = cv2.merge([outL, outA, outB])
  outputLab = np.uint8(outputLab)

  output = cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)

  output = output + 0.8 * output.std() * np.random.random(output.shape)
  # cv2.line(output, (450, 0), (450, 345), (0,0,0), thickness = 1, lineType=cv2.LINE_AA)
  #  cv2.imwrite("results/oldify_%d.jpg" % (random.randint(0, 10000)), output)
  return output


def alphablend(background, foreground):

  b, g, r, a = cv2.split(foreground)

  # Save the foregroung RGB content into a single object
  foreground = cv2.merge((b, g, r))

  # Save the alpha information into a single Mat
  alpha = cv2.merge((a, a, a))

  # Read background image
  background = cv2.imread("../data/images/backGroundLarge.jpg")

  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)
  alpha = alpha.astype(float)/255

  # Perform alpha blending
  foreground = cv2.multiply(alpha, foreground)
  background = cv2.multiply(1.0 - alpha, background)
  outImage = cv2.add(foreground, background)
  return outImage
