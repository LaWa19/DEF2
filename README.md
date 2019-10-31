# DEF2
Bike Speedcam

# Goal
This project is the result of the subject DEF 2 as part of the bachelor Applied Physics at the Delft University of Technology. For this course we had to design a speed measuring device to measure the speed of passing bicycles. To do so we decided to make use of a camera as it is a non intrusive way to measure speed, in comparison to for example pressure plates. 

# Method
AI is being used to detect a bike. By following the middle point of those bikes the speed can be determined over a certain distance. The measurement setup consists of a Raspberry Pi Zero with a camera, which uploads 20 second video's over wifi to a server. Using a python script on an off-site computer those videos are analized one by one to compute the speed of each cyclist.
