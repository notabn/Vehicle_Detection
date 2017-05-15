import cv2


vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  #print ('Read a new frame: ', success)
  count += 1
  if count > 550 :
      cv2.imwrite("tframe%d.jpg" % count, image)  # save frame as JPEG file
  if count > 560 :
     success = False
