# DEF2 - Bike Speedcam

# Goal
This project is the result of the course DEF 2 as part of the bachelor Applied Physics at the Delft University of Technology. For this course we had to design a speed measuring device to measure the speed of passing bicycles. To do so we decided to make use of a camera as it is a non intrusive way to measure speed, in comparison to for example pressure plates. 

# Method
AI is being used to detect a bike. By following the middle point of those bikes the speed can be determined over a certain distance. The measurement setup consists of a Raspberry Pi Zero with a camera, which uploads 20 second video's over wifi to a server. Using a python script on an off-site computer those videos are analized one by one to compute the speed of each cyclist.

# Acknowledgements
Model to recognise bikes
https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

SSIM value calculation
https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

MSE value
https://gist.github.com/gonzalo123/df3e43477f8627ecd1494d138eae03ae
