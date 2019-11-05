# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:23:23 2019

@author: Roeland
"""

#   let op regel 496=schijf,  staat uit 

# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import numpy as np
import datetime
import cv2
from copy import deepcopy
import glob
import os
from skimage.measure import compare_ssim
import time
import delPrullenbak as delprul


#%% de eerst keer wordt de tijd van start en de grote van de foto's verzonden door de stream en hier uitgelezen
def get_startup(paths):
    #file extentie
    extend="*.h264"
    print("wait for video...")
    #wacht totdat er video's aanwezig zijn
    while len(paths)<1:
        paths=[]
        paths = glob.glob(p+extend)
        time.sleep(10)
    #print(paths)
    #zet de tijd van nu om naar het goed format
    nu=datetime.datetime.now()
    nu=nu.strftime(tijdformat)
    nu=datetime.datetime.strptime(nu, tijdformat)
    
    #Zegt dat de kleinste tijd de tijd van nu is
    dtijden=nu
    teller=0
    poskleinDt=0
    #kijkt of er een kleinere tijd aanwezig is dan de tijd van nu in de video's
    for path in paths:
#        os.remove(path)
        tijden=path[len(p):]
        dateTijden=datetime.datetime.strptime(tijden[:-len(extend)+1],tijdformat)
        #slaat de positie van de kleinste tijd op van de paths lijst 
        if dateTijden<dtijden:
            dtijden=dateTijden
            poskleinDt=teller
        teller+=1	
    #de begin tijd van de stream
    global tstart
    
    #de opslag plaats van het snelheidsbestand
    global txtPath
    global txtP1
    
    #tvideo laat zien hoelaat de video begonnen is
    tvideo=paths[poskleinDt][len(p):-len(extend)+1]
    
    #slaat de tijd van de eerst video op in een txt file
    txtPath="snelheid_v_t.txt"
    txtfile=open(txtPath,"w")
    txtfile.write(str(tvideo)+"\n")
    txtfile.close() 
    
    txtP1 = "snelheden.txt"
    txt1 = open(txtP1, "w")
    txt1.write(str(tvideo+"\n"))
    txt1.close()
    #maakt het goede format aan van de tijd
    tvideo=datetime.datetime.strptime(tvideo, tijdformat)
    tstart=tvideo
    
    return p+paths[poskleinDt][len(p):],poskleinDt,paths,tvideo

#%% Mean Squared Error
def mse(i,j,bikeList,videoList,bestMSE,vgem):
    
    #vraagt de positie op, om twee fietsers te vergelijken
    midXOld,midYOld=int(bikeList[i][-1][0]),int(bikeList[i][-1][1])
    midXNew,midYNew=int(bikeList[j][-1][0]),int(bikeList[j][-1][1])
    
    #zet de grote van een vierkant
    groteMSE=int(10/200*pixelWidth)
    #print(groteMSE)
    
    if pixelWidth*ratioImg-(groteMSE+midYNew)<0:
        groteMSE=int(pixelWidth*ratioImg-(midYNew))
    if pixelWidth*ratioImg-(groteMSE+midYOld)<0:
        groteMSE=int(pixelWidth*ratioImg-(midYOld))
    if midYOld-groteMSE<0:
        groteMSE=int(midYOld) 
    if midYNew-groteMSE<0:
        groteMSE=int(midYNew)
    #print(groteMSE)
        
    
    #maakt twee afbeelding rond het midden in de vorm van een vierkant van elke fietser
    imageA=videoList[-1][midYNew-groteMSE:midYNew+groteMSE, midXNew-groteMSE:midXNew+groteMSE]
    imageB=videoList[bikeList[i][-1][5]-videoListRemoved][midYOld-groteMSE:midYOld+groteMSE, midXOld-groteMSE:midXOld+groteMSE]
    
    #laat de foto's zien
#    cv2.imshow("A",imageA)
#    cv2.imshow("B",imageB)
        
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    
    #Een lager score geeft een meer gelijkheid van de afbeeldingen
    score=1-score
    
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    #geeft de laatste waarde terug, de best lijkende foto
    if err*score<bestMSE[1]:
        bestMSE=[i,err*score]
        
    
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return bestMSE
#%%
def printSnelheid(snelheden,fietsersgehad,fietsbuffer,fietsermetv0):
    #opent het bestand met snelheden 
    txtfile=open(txtPath,"a")
    txt1 = open(txtP1, "a")
    for it in range(0,len(bikeList)-amountBikeFrame-fietsbuffer):
        ij=0
        #laat fietsers met een snelheid van 0 buiten beschouwing
        if len(bikeList[ij])>1:
            
            #neemt begin en eind tijd en de gemiddelde snelheid
            amountFrame1=bikeList[ij][0][2]
            amountFrame2,vgem2=bikeList[ij][-1][2],bikeList[ij][-1][4]
            #bepaald de gemiddelde hoogte waar de fietser heeft gereden 
            gemEndY=0
            for iy in range(len(bikeList[ij])):
                gemEndY+=bikeList[ij][iy][3]/len(bikeList[ij])
                
            #berekening van tijd
            t1=datetime.datetime.strptime(amountFrame1, tijdformat)
            t2=datetime.datetime.strptime(amountFrame2, tijdformat)
            t1=t1-tstart
            t2=t2-tstart
            t=(t2+t1).total_seconds()/2
            
            #correctie van snelheid voor vervorming beeld
            s=(s1-s0)*(1-(gemEndY/(pixelWidth*ratioImg)))+s0
            alpha=pixelWidth/s
            v=vgem2*fps1/alpha
            print("fietser",fietsersgehad-fietsermetv0,": v =",v,"m/s = ",v*3.6,"km/h")
            snelheden.append(v)
            txtfile.write(str(v)+","+str(t)+","+str(v*3.6)+"\n")
            #Readable txt file
            txt1.write("fietser %s : v= " % fietsersgehad)
            txt1.write("%s m/s =" % np.round(v,decimals=1))
            txt1.write("%s km/h \n" % np.round(v*3.6,decimals =1))

        #telt het aantal fietsers met v=0.0
        else:
            fietsermetv0+=1
            
        #zorgt ervoor dat de fietser niet meer mee wordt genomen in de while loop
        fietsersgehad+=1
        bikeList.pop(0)
        
        
        
    #sluit snelheden bestand
    txtfile.close()
    txt1.close()
    return snelheden,fietsersgehad,fietsermetv0

#%%
def fietskoppelen(bikeList):
    #is de nieuwe fietsers
    j=len(bikeList)-1
    
    i=0
    #is de loop van de oude fietsers 
    #i<j zorgt ervoor dat er geen overlap is
   
    #zet de tijd van de nieuwe fietsers om in het goede format
    fietserTijdNew=datetime.datetime.strptime(bikeList[j][-1][2], tijdformat)
    
    #bestMSE wordt gebruikt voor de vergelijing van fietsers
    bestMSE=[-1,np.inf]
    while i<j:

        #zet de tijd van de oude fietsers om in het goede format
        fietserTijdOld=datetime.datetime.strptime(bikeList[i][-1][2], tijdformat)
        
        #neemt het verschil in frames van de oudde fietsers en de nieuwe fietser
        dframeTijd=fietserTijdNew-fietserTijdOld
        dframe=fps1*(dframeTijd.total_seconds())
        #print("dframe:",dframe)
        #fietser mogen niet in hetzelfde frame worden vergeleken
        if dframe>0:
            
            #kijken of op basis van snelheid de fietser aan de fietser van het vorge frame kan worden gekoppelt
            if len(bikeList[i])>=2:
                vgem=bikeList[i][-1][4]
                
                #berekent mogelijk positie van de al bekende fiets
                mogelijkpos=bikeList[i][-1][0]+(vgem)*(dframe)
                
                #kijkt of de mogelijke positie binnen de rechthoek valt
                if mogelijkpos>0 and mogelijkpos<pixelWidth:
                    
                    #berekent het verschil in afstand tussen de mogelijke positie en de fiets in dit frame
                    dSFrames=abs(mogelijkpos-bikeList[j][0][0])   
                    
                    #kijkt of het mogelijke verschil bij de fietser past + een onzekerheid in het verschil in frames
                    # zo ja wordt er gekeken bij welke fiets de MSE keer SSIM-waarde het kleinst is
                    if dSFrames< afstandvooruitpixels+(2*200/pixelWidth*dframe) and  abs(bikeList[i][-1][1]-bikeList[j][0][1])<afstandopzijpixels+dframe*200/pixelWidth:
                        bestMSE=mse(i,j,bikeList,videoList,bestMSE,vgem)
                        if bestMSE[1]>9000:
                            bestMSE=[-1,np.inf]
                        
            # fietsers met nog geen vastgestelde snelheid
            if len(bikeList[i])<2:
                #Het maximale verschil dat er tussen de fietser in frames mag zijn
                maxdframe=fps1/((rechtsrechthoek-linksrechthoek)/(20/3.6*alpha0))
                if dframe<maxdframe:
                    #kijkt welke fietser het beste past met een maximium verschil
                    bestMSE=mse(i,j,bikeList,videoList,bestMSE,vgem=0)
                    if bestMSE[1]>7500:
                        bestMSE=[-1,np.inf]
                       
                    

        #aan het einde van de while loop wordt van de fietsers gekeken of de fietser dezelfde fiesters is
        #zo ja dan worden de fietser gekoppelt en op de andere plek verwijderd
        #zo nee dan staat de fietser apart in de bikeList
        if bestMSE[1]!=np.inf and i==j-1:
            pos=bestMSE[0]
            tbeginFiets=datetime.datetime.strptime(bikeList[pos][0][2], tijdformat)
            vgem=(bikeList[j][0][0]-bikeList[pos][0][0])/(fps1*(absTijd-tbeginFiets).total_seconds())
            bikeList[pos].append([bikeList[j][0][0],bikeList[j][0][1],bikeList[j][0][2],bikeList[j][0][3],vgem,bikeList[j][0][5]])
            
            #slaat de foto van de fietser op 
            midXNew,midYNew=int(bikeList[j][-1][0]),int(bikeList[j][-1][1])
            groteMSE=int(10/200*pixelWidth)
            imageA=videoList[-1][midYNew-groteMSE:midYNew+groteMSE, midXNew-groteMSE:midXNew+groteMSE]
            cv2.imwrite("fietsers/fietser_"+str(pos+fietsersgehad+1-fietsermetv0)+"_"+str(len(bikeList[pos])+fietsersgehad-1)+"_"+str(int(bestMSE[1]))+".png",imageA)
            
            #vewijdert de fietser uit bikeList
            bikeList.pop(j)
#            j=j-1

        i=i+1

    return bikeList
#%%
def herken(bikeList,videoFrame,frame,tvideo,snelheden,fietsersgehad,fietsermetv0):
    amountBikeFrame=0
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        
		# extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence > confidence1:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
			# draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
            
            #berekent midden van de fietser
            middenX,middenY=((endX+startX)/2),((endY+startY)/2)

            #herkenning binnen een bepaalde rechthoek, zorgt er voor dat de fiets volledig in beeld is
            if middenX>(linksrechthoek) and middenX<(rechtsrechthoek):
                if (CLASSES[idx])=="bicycle" or (CLASSES[idx])=="motorbike":
                    #tekent een rechthoek om de fietser
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                				COLORS[idx], 2)
                    #tekent een cirkel in het midden van de fietser
                    cv2.circle(frame, (int(middenX), int(middenY)),
                               3, (0, 255, 0), thickness=2, lineType=8, shift=0)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
        				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                   
                    #telt aantal fietsen per frame
                    amountBikeFrame=amountBikeFrame+1
                    #geeft de fietser een snelheid
                    vgem=0
                    
                    # datetime(year, month, day, hour, minute, second, microsecond)
                    absTijdStr=absTijd.strftime(tijdformat)
                    #slaat de lokatie van de elke fietser met frame nummer, absolutetijd , plek in de lijst met alle beelden
                    bikeList.append([[middenX,middenY,absTijdStr,endY,vgem,plekinVideoList]])
                    bikeList=fietskoppelen(bikeList)
        
                    #slaat de snelheid op als de fietser uit beeld is
                    # de buffer zorgtervoor dat de fietser tijdelijk uit beeld kan zijn
                    if len(bikeList)-amountBikeFrame-fietsbuffer>0:
                        snelheden,fietsersgehad,fietsermetv0=printSnelheid(snelheden,fietsersgehad,fietsbuffer,fietsermetv0)
    return bikeList,amountBikeFrame,frame,snelheden,fietsersgehad,fietsermetv0


#%%
def mse8uur(imageA,imageB,nextVid):
    
    # convert the images to grayscale
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    #geeft de laatste waarde terug, de best lijkende foto
    if err>200:
        nextVid=False
    
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return nextVid

#%%
def newVid(paths,p,stopLoop,poskleinDt):
    
    #pakt de begintijd van de laatste video
    laatstetijd=paths[poskleinDt][len(p):]
    
    #verwijdert de laatste video
    os.remove(p+laatstetijd)
    
    #video extentie
    extend="*.h264"
    laatstetijd=laatstetijd[:-len(extend)+1]
    
    #verwijdert het path van de laatste video
    paths.pop(poskleinDt)

    #Wacht totdat er een video is 
    while len(paths)<1:
        print("wait for video...")
        paths=[]
        paths = glob.glob(p+extend)
        key = cv2.waitKey(1) & 0xFF
    	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            stopLoop=True
            break 

    if stopLoop==False:
        
        #zet de tijd van nu om naar het goed format
        nu=datetime.datetime.now()
        nu=nu.strftime(tijdformat)
        nu=datetime.datetime.strptime(nu, tijdformat)
        laatstetijd=datetime.datetime.strptime(laatstetijd, tijdformat)
        
        #Zegt dat de kleinste tijd de tijd van nu is
        dtijden=nu
        teller=0
        poskleinDt=0
        #kijkt of er een kleinere tijd aanwezig is dan de tijd van nu in de video's
        for path in paths:
    #        os.remove(path)
            tijden=path[len(p):]
            dateTijden=datetime.datetime.strptime(tijden[:-len(extend)+1],tijdformat)
            #slaat de positie van de kleinste tijd op van de paths lijst 
            if dateTijden<dtijden and dateTijden >laatstetijd:
                dtijden=dateTijden
                poskleinDt=teller
            teller+=1	
        #de begin tijd van de video
        tvideo=paths[poskleinDt][len(p):-len(extend)+1]
        tvideo=datetime.datetime.strptime(tvideo, tijdformat)
        video_src=p+paths[poskleinDt][len(p):]
        vs = cv2.VideoCapture(video_src)
        _,frame = vs.read()
            
        #wacht totdat de video volledig is upgeload
        while (type(frame) == type(None)):
            #print("wait.. NoneType")
            vs.release()
            vs = cv2.VideoCapture(video_src)
            _,frame = vs.read()
        
        nextVid=False
        if int(tvideo.strftime("%H"))>= 14 or int(tvideo.strftime("%H"))<= 6: 
            vs.release()
            nextVid=True
            nu=datetime.datetime.now()
            vs2 = cv2.VideoCapture(video_src)
            _, frame_oud = vs2.read()
            i = 0
            while (True):
                key = cv2.waitKey(1) & 0xFF
            	# if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    stopLoop=True
                    break 
                _, frameCheck = vs2.read()
                if type(frameCheck)==type(None):
                    break   
                if i/20 == int(i/20):
                    # Take each frame
                    
                    nextVid=mse8uur(frame_oud,frameCheck,nextVid)
                    if nextVid==False:
                        break
                    frame_oud=frameCheck
                i+=1
            #print(datetime.datetime.now()-nu)
    #stopt de while loop
    if stopLoop==True:
        video_src=p
        poskleinDt=0
        tvideo=""
        nextVid=False


#    print(laatstetijd,starttijden[poskleinDt])
    return video_src,stopLoop,poskleinDt,paths,tvideo,nextVid
#%%
#vraagt start tijd van script op
tstartscript=datetime.datetime.now()

#verwijdert aan gemaakte foto's van de vorige keer
plek="fietsers/fietser"
extend="*.png"
paths = glob.glob(plek+extend)
#print(paths)
for path in paths:
    os.remove(path)
    
# afstand van links naar rechts van fietspad in meters
s1=11.4 #afstand in het bovenaan
s=31*0.3#afstand in het midden
s0=8.1#afstand in het onderin

pixelWidth=1080 #gewenste breedte van de foto
ratioImg=0.5625 #ratio hoogte en breedte image
alpha0=pixelWidth/s #van meter naar pixels
bikeList=[] # wordt de lijst met alle fietsers
snelheden=[] # lijst met snelheden
fietsersgehad=0 #fietsers die al uit beeld zijn
videoList=[]#alle afbeeldingen, de hele video

#opslag plaats van model en percentage van de minimale overeenkomst
prototxt, model,confidence1="MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel",0.3

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

#maakt random kleur aan per class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

#neemt de mogelijkheid weg dat een fiets even niet herkent wordt maar wel aanwezig is.
fietsbuffer=2

#fietsers met een snelheid=0, worden bij het uitlezen niet meegenomen
fietsermetv0=0

#plek waar de video's zijn opgeslagen
paths=[]
p="C:/Users/mccoy/stack/streamVid/"

#Het normale format van datetime heeft ":", dit kan niet worden opgeslagen in een bestandsnaam
tijdformat="%Y-%m-%d %H-%M-%S-%f"

#leegt de prullenbak van de netwerkopslag
prul=delprul.LaPrul()

#vraagt de video met de kleinste absolute tijd aan
video_src,poskleinDt,paths,tvideo=get_startup(paths)

#pakt video en eerste frame
vs = cv2.VideoCapture(video_src)
_,frame = vs.read()

#wacht totdat de video volledig is upgeload
while (type(frame) == type(None)):
    #print("wait.. NoneType")
    vs.release()
    vs = cv2.VideoCapture(video_src)
    _,frame = vs.read()
    
#    time.sleep(10)
    
#laat video los en begint vanaf het begin, vraagt het aantal fps op van de video
vs.release()
vs = cv2.VideoCapture(video_src)
fps1=vs.get(cv2.CAP_PROP_FPS)

#wordt gebruikt voor de rechthoek om fietsers te koppelen
snelheidvooruit=50 #max snelheid vooruit km/h
snelheidopzij= 30#max snelheid opzij km/h

#hardste snelheid naar afstand in pixels
afstandvooruitpixels=snelheidvooruit/3.6/fps1*alpha0
afstandopzijpixels=snelheidopzij/3.6/fps1*alpha0

#rechthoek binnen video afbakkenen
linksrechthoek=70/200*pixelWidth
rechtsrechthoek=130/200*pixelWidth

#aanmaak variable
videosinPrul=0 #hoeveel video's in prubbelbak netwerkopslag
videoFrame=0 #het aantal beeld dat zijn geweest in de video
global absTijd #de absolute tijd waar het script zich bevindt
plekinVideoList=0 #plek waar video's in de videoList worden opgeslagen
stopLoop=False #Stopt de While loop: stopLoop==True
nu1=datetime.datetime.now()
videoListRemoved=0
#%%
while True:

	 #grab the frame from the threaded video stream
    _,frame = vs.read()
    #kijkt of er een foto is.
    if (type(frame) == type(None)):
        #print("videoTijd",datetime.datetime.now()-nu1)
        nu1=datetime.datetime.now()
        #laat de video los
        vs.release()
        nextVid=True
        #pakt de nieuwe video met de kleinste absolute tijd
        while nextVid==True:
            video_src,stopLoop,poskleinDt,paths,tvideo,nextVid=newVid(paths,p,stopLoop,poskleinDt)
            #telt hoeveel video's er in de prullenvak zitten
            videosinPrul+=1
            #met stopLoop kan alles gestopt worden
            if stopLoop==True:
                break
            #leegt de prullenbak van de netwerkopslag
            if videosinPrul>20:
                if len(bikeList)>0:
                    for rm in range(videoListRemoved,bikeList[0][0][5]-1):
                        videoList.pop(0)
                        videoListRemoved+=1
                    prul.prul()
                    videosinPrul=0
        
        #met stopLoop kan alles gestopt worden
        if stopLoop==True:
            break

        #pakt nieuwe video
        vs = cv2.VideoCapture(video_src)
        videoFrame=0
        _,frame = vs.read()
            
        #leest het aantal fps uit van de video
        fps1=vs.get(cv2.CAP_PROP_FPS)
            
    
    #rekent het aantal frames van de video om naar een datetime format en wordt bij de absolute tijd opgeteld
    fps1tijd=datetime.timedelta(seconds=videoFrame/fps1)
    absTijd=tvideo+fps1tijd
    
    #schaalt de video naar de gewenste grote
    frame = cv2.resize(frame, (pixelWidth,int(pixelWidth*ratioImg)))
    
    #slaat alle frames op in een list
    videoList.append(deepcopy(frame))
	# grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
    nu=datetime.datetime.now()
	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

    nu=datetime.datetime.now()
    # leest de detections uit binnen een bepaalde rechthoek en geeft de middelpunten van alle fietsers terug met het frame nummer
    bikeList,amountBikeFrame,frame,snelheden,fietsersgehad,fietsermetv0=herken(bikeList,videoFrame,frame,tvideo,snelheden,fietsersgehad,fietsermetv0)


     # show the output frame      
    cv2.imshow("Server", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break 
    
    #video en videoplek worden verhoogd voor het volgende frame
    videoFrame+=1
    plekinVideoList+=1


#%%
#stopt de chromedriver
prul.teardown()

amountBikeFrame=0
#print, slaat op snelheid en zet video weg als de fietser uit beeld is 
snelheden,fietsersgehad,fietsermetv0=printSnelheid(snelheden,fietsersgehad,0,fietsermetv0)

#sluit alle show windows
cv2.destroyAllWindows()

print(snelheden)

#print tijd hoelang het script bezig was
#print("tijd: ",datetime.datetime.now()-tstartscript)
