import shutil
import glob
import time
import datetime

extend="*.h264"
saveFile="/home/pi/Documents/video/done/"

i=0
tijdformat="%Y-%m-%d %H-%M-%S-%f"
while True:
	i+=1
	starttijden=[]
	paths=[]
	while len(paths)<1:
		paths=[]
		paths=glob.glob(saveFile+extend)
			
		if len(paths)<1:
			print("wait...")
			time.sleep(20)

	nu=datetime.datetime.now()
	nu=nu.strftime(tijdformat)
	nu = datetime.datetime.strptime(nu,tijdformat)
	dtijden=nu
	poskleinDt=0
	teller=0
	for path in paths:
		print(path)
		tijden=path[len(saveFile):]	
#		starttijden.append(tijden)
		dateTijden=datetime.datetime.strptime(tijden[:-len(extend)+1],tijdformat)
		if dateTijden<dtijden:
			dtijden=dateTijden
			poskleinDt=teller
		teller+=1	
	shutil.move(saveFile+paths[poskleinDt][len(saveFile):],"/mnt/dav/streamVid/"+paths[poskleinDt][len(saveFile):])
	print("Move Video")


