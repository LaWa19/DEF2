import picamera
import datetime
import shutil
import os
import glob

extend="*.h264"
saveFile="/home/pi/Documents/video/"
paths=[]
paths=glob.glob(saveFile+"done/"+extend)
for path in paths:
	os.remove(path)
paths=[]
paths=glob.glob(saveFile+extend)
for path in paths:
	os.remove(path)
print("haalo")
camera= picamera.PiCamera()
pixelWidth=1080
imgRatio=0.5625

resolutie=(pixelWidth,int(pixelWidth*imgRatio))
fps=30
tijdformat="%Y-%m-%d %H-%M-%S-%f"
camera.resolution=resolutie
camera.framerate=fps
opneemtijd=20

i=0
while True:
	i+=1
	nu=datetime.datetime.now()
	nu=nu.strftime(tijdformat)

	camera.start_recording(saveFile+str(nu)+".h264")
	camera.wait_recording(opneemtijd)
	camera.stop_recording()
	shutil.move(saveFile+str(nu)+".h264",saveFile+"done/"+str(nu)+".h264")
	print("save Video")



