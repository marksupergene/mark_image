import cv2, json, dlib, os, random
import numpy as np
import api_practice.utils as utils
import api_practice.image_utils as image_utils
import api_practice.video_utils as video_utils
import api_practice.faceBlendCommon as fbc
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

PREDICTOR_PATH = 'common/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Read all jpg image paths in folder.
def readImagePaths(path):
    # Create array of array of images.
    imagePaths = []
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:
            print(filePath)

            # Add to array of images
            imagePaths.append(os.path.join(path, filePath))
    return imagePaths

@api_view(['POST'])
def mona_lisa(request): # Mona Lisa museum painting
  # dirName = "src/mona_lisa"

  # Read all images
  # imagePaths = readImagePaths(dirName)

  # Read two images
  data = utils.parseRequest(request)
  src_url = data['src_url']
  dst_url = data['dst_url']

  im1 = utils.url_to_image(src_url)
  im2 = utils.url_to_image(dst_url)

  imagePaths = [im1, im2]

  if len(imagePaths) == 0:
    print('No images found with extension jpg or jpeg')
    sys.exit(0)

  # Read images and perform landmark detection.
  images = []
  allPoints = []

  for imagePath in imagePaths:
    im = imagePath
    if im is None:
      print("image:{} not read properly".format(imagePath))
    else:
        points = fbc.getLandmarks(
            faceDetector, landmarkDetector, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if len(points) > 0:
          allPoints.append(points)

          im = np.float32(im)/255.0
          images.append(im)
        else:
          print("Couldn't detect face landmarks")

  if len(images) == 0:
    print("No images found")
    sys.exit(0)

  # Dimensions of output image
  w = 600
  h = 600

  # 8 Boundary points for Delaunay Triangulation
  boundaryPts = fbc.getEightBoundaryPoints(h, w)

  numImages = len(imagePaths)
  numLandmarks = len(allPoints[0])

  # Variables to store normalized images and points.
  imagesNorm = []
  pointsNorm = []

  # Initialize location of average points to 0s
  pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)

  # Warp images and trasnform landmarks to output coordinate system,
  # and find average of transformed landmarks.
  for i, img in enumerate(images):

    points = allPoints[i]
    points = np.array(points)

    img, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)

    # Calculate average landmark locations
    pointsAvg = pointsAvg + (points / (1.0*numImages))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.concatenate((points, boundaryPts), axis=0)

    pointsNorm.append(points)
    imagesNorm.append(img)

  # Append boundary points to average points.
  pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)

  # Delaunay triangulation
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Output image
  output = np.zeros((h, w, 3), dtype=np.float)

  # Warp input images to average image landmarks
  for i in range(0, numImages):

    imWarp = fbc.warpImage(
        imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)

    # Add image intensities for averaging
    output = output + imWarp

  # Divide by numImages to get average
  output = output / (1.0*numImages)

  # print(output)
  output = np.uint8(output * 255)

  # Read source image.
  im_src = np.copy(output)
#   im_src = cv2.imread('images/grad_mark.jpg')
  # Four corners of the book in source image
  pts_src = np.array(
      [[0, 0], [599, 0], [599, 599], [0, 599]], dtype=float)

  # Read destination image.
  im_dst = cv2.imread('src/museum.jpg')
  # Four corners of the book in destination image.
  pts_dst = np.array(
      [[379, 153], [689, 137], [696, 483], [378, 477]], dtype=float)

  # Calculate Homography
  h, status = cv2.findHomography(pts_src, pts_dst)

  # Warp source image to destination based on homography
  im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

  # Create the basic black image
  mask = np.zeros(im_dst.shape, dtype="uint8")

  # Draw a white, filled rectangle on the mask image
  # cv2.rectangle(mask, (44, 357), (720, 740), (255, 255, 255), -1)

  pts = np.array([[379, 153], [689, 137], [696, 483], [378, 477]], np.int32)
  pts = pts.reshape((-1, 1, 2))
  cv2.fillPoly(mask, [pts], (255, 255, 255))

  # Display images
  mask_inv = cv2.bitwise_not(mask)
  maskedDst = cv2.bitwise_and(im_dst, mask_inv)
  maskedOut = cv2.bitwise_and(im_out, mask)
  masked = cv2.add(maskedDst, maskedOut)
  url = utils.image_to_url("results/face_average.jpg", masked)
  return Response({"image_url": url})

@api_view(['POST'])
def friends_morph(request): # Face morph to FRIENDS character

  # Read two images
  data = utils.parseRequest(request)
  src_url = data['src_url']
  dst_url = data['dst_url']

  im1 = utils.url_to_image(src_url)
  im2 = utils.url_to_image(dst_url)

  bg = cv2.imread("src/friends_frame.png")

  # Detect landmarks in both images.
  points1 = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
  points2 = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))

  points1 = np.array(points1)
  points2 = np.array(points2)

  # Convert image to floating point in the range 0 to 1
  im1 = np.float32(im1)/255.0
  im2 = np.float32(im2)/255.0

  # Dimensions of output image
  h = 400
  w = 300

  # Normalize image to output coordinates.
  imNorm1, points1 = fbc.normalizeImagesAndLandmarks((h, w), im1, points1)
  imNorm2, points2 = fbc.normalizeImagesAndLandmarks((h, w), im2, points2)

  # Calculate average points. Will be used for Delaunay triangulation.
  pointsAvg = (points1 + points2)/2.0

  # 8 Boundary points for Delaunay Triangulation
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  points1 = np.concatenate((points1, boundaryPoints), axis=0)
  points2 = np.concatenate((points2, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)

  # Calculate Delaunay triangulation.
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Start animation.
  alpha = 0

  video = cv2.VideoWriter('results/face_morph/face_average.avi', -1,
                          10, (bg.shape[1], bg.shape[0]))

  frames = []

  while alpha < 1:
      # Compute landmark points based on morphing parameter alpha
      pointsMorph = (1 - alpha) * points1 + alpha * points2

      # Warp images such that normalized points line up with morphed points.
      imOut1 = fbc.warpImage(imNorm1, points1, pointsMorph.tolist(), dt)
      imOut2 = fbc.warpImage(imNorm2, points2, pointsMorph.tolist(), dt)

      # Blend warped images based on morphing parameter alpha
      imMorph = (1 - alpha) * imOut1 + alpha * imOut2

      imMorph = np.uint8(imMorph * 255)

      x_offset = int((bg.shape[1]/2) - (imMorph.shape[1]/2))
      y_offset = int((bg.shape[0]/2) - (imMorph.shape[0]/2))

      bg[y_offset:y_offset + imMorph.shape[0],
          x_offset:x_offset + imMorph.shape[1]] = imMorph

      video.write(bg)

      alpha += 0.025
  video.release()

  path = "results/face_morph/face_average.avi"
  # video_utils.video_write(path, 10, (imMorph.shape[1], imMorph.shape[0]), frames)
  url = utils.video_to_url(path)
  return Response({"video_url": url})
