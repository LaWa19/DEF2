# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:39:53 2019

@author: Roeland
"""
import matplotlib.pyplot as plt
import numpy as np
txtPath="snelheid_v_t.txt"

#txtfile=open("snelheid_v_t.txt","w")
#txtfile.close()
#txtfile=open(txtPath,"a")
#for i in range(3):
#    v=9*i
#    t1=i**2
#    txtfile.write(str(v)+","+str(t1)+","+str(v*3.6)+"\n")
#txtfile.close()


txtfile=open(txtPath,"r")
txt =txtfile.read()

vL=[]
tL=[]
vkmhL=[]
vR=[]
tR=[]
vkmhR=[]
txt=(txt.split("\n"))

for i in range(1,len(txt)):
    tekst=txt[i].split(",")
    try:
        mogV=round(float(tekst[0]),2)
        if mogV<0:
            vL.append(round(float(tekst[0]),2))
            tL.append(round(float(tekst[1]),2))
            vkmhL.append(round(float(tekst[2]),2))
        else:
            vR.append(round(float(tekst[0]),2))
            tR.append(round(float(tekst[1]),2))
            vkmhR.append(round(float(tekst[2]),2))
            
    except:
        z=1
v=vL+vR
t=tL+tR

#plt.plot(v,t)

plt.hist(v, bins = 10)
if len(vR)>0:
    plt.plot([sum(vR)/len(vR),sum(vR)/len(vR)],[0,max(len(vR),len(vL))/np.sqrt(2)])
if len(vL)>0: 
    plt.plot([sum(vL)/len(vL),sum(vL)/len(vL)],[0,max(len(vR),len(vL))/np.sqrt(2)])

plt.savefig("snelheidHist.png")
plt.close()

#plt.hist(t,bins=10)

tmax=0
tmin=np.inf
for i in t:
    tmin=min(tmin,i)
    tmax=max(i,tmax) 
width=0.01*tmax
plt.bar(t,v,width=width)
plt.plot([tmin,tmax],[0,0],color='r')
plt.title("Vanaf: "+txt[0])
plt.xlabel("tijd (s)")
plt.ylabel("snelheid (m/s)")
plt.show()    
plt.savefig("snelheidTijd.png")
plt.close()
    
