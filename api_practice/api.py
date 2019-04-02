import cv2, json, dlib, os, random
import numpy as np
import api_practice.utils as utils
import api_practice.image_utils as image_utils
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

def noisy(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * (gauss/2)
    return noisy

@api_view(['POST'])
def ig_filter(request):

    data = utils.parseRequest(request)
    url = data['image_url']

    src = utils.url_to_image(url)

    # Median blur
    kernelSize = 5
    dst = cv2.medianBlur(src, kernelSize)

    dst2 = cv2.edgePreservingFilter(dst, flags=1, sigma_s=60, sigma_r=0.4)

    saturationScale = 1.5
    hsvImage = cv2.cvtColor(dst2, cv2.COLOR_BGR2HSV)
    hsvImage = np.float32(hsvImage)
    H, S, V = cv2.split(hsvImage)
    S = np.clip(S * saturationScale, 0, 255)
    hsvImage = np.uint8(cv2.merge([H, S, V]))
    imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

    url = utils.image_to_url("results/ig_filter.jpg", imSat)
    return Response({"image_url": url})

@api_view(['POST'])
def old_filter(request):

    data = utils.parseRequest(request)
    url = data['image_url']

    original = utils.url_to_image(url)

    b_channel, g_channel, r_channel = cv2.split(original)
    # creating a dummy alpha channel image.
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
    fin_orig = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    scaleDown = cv2.resize(fin_orig, None, fx=0.7, fy=0.7,
                        interpolation=cv2.INTER_LINEAR)
    dst = np.copy(original)
    dst = cv2.resize(original, None, fx=0.7, fy=0.7,
                    interpolation=cv2.INTER_LINEAR)

    src = cv2.imread('src/farm.jpg')
    output = np.copy(scaleDown)

    noise_img = noisy(scaleDown)
    noise_img = noise_img.astype('uint8')

    b, g, r, a = cv2.split(noise_img)
    foreground = cv2.merge((b, g, r))
    alpha = cv2.merge((a, a, a))
    foreground = foreground.astype(float)
    dst = dst.astype(float)
    alpha = alpha.astype(float)/255
    foreground = cv2.multiply(alpha, foreground)
    dst = cv2.multiply(1.0 - alpha, dst)
    final = cv2.add(foreground, dst)
    final = final.astype('uint8')

    output = np.copy(dst)
    output = output.astype('uint8')

    srcLab = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
    dstLab = np.float32(cv2.cvtColor(final, cv2.COLOR_BGR2LAB))
    outputLab = np.float32(cv2.cvtColor(output, cv2.COLOR_BGR2LAB))

    srcL, srcA, srcB = cv2.split(srcLab)
    dstL, dstA, dstB = cv2.split(dstLab)
    outL, outA, outB = cv2.split(outputLab)

    outL = dstL - dstL.mean()
    outA = dstA - dstA.mean()
    outB = dstB - dstB.mean()

    outL *= srcL.std() / dstL.std()
    outA *= srcA.std() / dstA.std()
    outB *= srcB.std() / dstB.std()

    outL = outL + srcL.mean()
    outA = outA + srcA.mean()
    outB = outB + srcB.mean()

    outL = np.clip(outL, 0, 255)
    outA = np.clip(outA, 0, 255)
    outB = np.clip(outB, 0, 255)

    outputLab = cv2.merge([outL, outA, outB])
    outputLab = np.uint8(outputLab)

    output = cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)
    gaus = cv2.GaussianBlur(output, (5, 5), 0, 0)

    url = utils.image_to_url("results/old_filter.jpg", gaus)
    return Response({"image_url": url})
