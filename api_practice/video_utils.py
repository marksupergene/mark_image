import cv2
import api_practice.utils as utils

def video_write(path, fps, size, frames):
  video = cv2.VideoWriter(path,
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
  # video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', 'e', 'g'), fps, size)

  for frame in frames:
    video.write(frame)
  # print(len(frames))
  video.release()
  return video
