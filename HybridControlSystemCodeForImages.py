#_______________________________________________________________________________________________________________________
# *** VIDEO FEED AND PROGRAMME SETUP INCLUDING COMMUNICATION WITH MICROCONTROLLER ***
#_______________________________________________________________________________________________________________________

#Import all required libraries:
import cv2
import time
import math
import numpy as np
import serial
from ultralytics import YOLO

#_______________________________________________________________________________________________________________________
#Link computer to Arduino microcontroller:

#try:
    #serial1: serial.Serial(port = 'COM5', baudrate = 9600, timeout = 1); #Sets up serial communication with Arduino microcontroller
    #time.sleep(2); #Lets the Arduino reset before starting communication to prevent errors

#except:
    #print("No Arduino serial port found") #If no port is found notify the user+

#_______________________________________________________________________________________________________________________
#Import an image to run computer vision on:

Image = cv2.imread("C:/Users/james/Desktop/Dissertation 2025/Photos to test perspective based control on/Photos for testing/Photo_2026-04-12 18_23_54_176.JPG")

#_______________________________________________________________________________________________________________________
# *** MACHINE VISION ROBOT PIPE NAVIGATION CODE SECTION, USING AN ELLIPSE AND PERSPECTIVE CAST FOR PIPE NAVIGATION ***
#_______________________________________________________________________________________________________________________
#Modifying the images for use by the computer:

#Applying 'smoothing' to the video feed image using a Gaussian Blur filter
gaussianImage = cv2.GaussianBlur(Image, (5,5), 0)

#Applying a grayscal filter to the image to allow binary thresholding to be applied later
grayImage = cv2.cvtColor(gaussianImage, cv2.COLOR_BGR2GRAY)

#Using binary thresholding to create a high contrast black & white image that contours can easily be drawn onto
T, binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #For binary thresholding 0 is black and 255 is white here we are using a threshold level of 127 which will be tuned as neccesary

#Line of code for bug fixing initial image for trace creation:
#    cv2.imshow('Video Feed', binaryImage)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cv2.destroyAllWindows()
#print("Program Ended")

#_______________________________________________________________________________________________________________________
#Tracing the ellipse onto the colour image and assign output information to variables:

#Finding contours on the binary image
allContours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #May need to change RETR_TREE to RETR_EXTERNAL

#Sort contours based on area to
if allContours:
    #Use max() with cv2.contourArea to find the most significant trace
    contour = max(allContours, key=cv2.contourArea)
else:
    contour = None
    print("Warning: No contours detected. Check your threshold level.")

#Tracing an ellipse onto the contour created then pulling out the readouts provided
ellipse1 = cv2.fitEllipse(contour) #Fitting the ellipse
(centerCoordinates), (axesLengths), angle = ellipse1 #Pulling out ellipse data
xLocation = int(centerCoordinates[0]) #X-axis length
yLocation = int(centerCoordinates[1]) #Y-axis length
minorAxisLength = int(axesLengths[0]) #Length of major axis
majorAxisLength = int(axesLengths[1]) #Length of minor axis

#The above is correct and provides required information to my knowledge

#_______________________________________________________________________________________________________________________
#Providing a readout of the axes lengths (this method may now work but further testing is required):
#The ellipse drawing bit should be ok

if angle <= 45:
    height = majorAxisLength
    width = minorAxisLength
else:
    height = minorAxisLength
    width = majorAxisLength
#Print out the above values as a list
print("height:", height, "width", width, "angle:", angle)

#Draw the ellipse onto the camera feed
cv2.ellipse(Image, ellipse1, (120,0,120), 3)

#_______________________________________________________________________________________________________________________

#This section bellow has the issue of the axes changing unexpectedly which will cause problems for a robots control inputs: It may now have been rectified
#(06/03/2026):

#Find half axis lengths as the function gives the width and height of ellipse bounding box
halfMajorLength = int(axesLengths[1] / 2)
halfMinorLength = int(axesLengths[0] / 2)

#Convert angle to radians as python expects sin/cos inputs to be in this form and adjust for OpenCV's coordinate system
#Adjust for the way the function measures angles (switching of minor and major axes) and convert to radians
#NOTE: openCV measures angles clockwise from the x-axis
angleMajor = math.radians(angle)
angleMinor = math.radians(angle + 90)

#Draw major axis onto video feed and assign colour red with thickness 2
X1 = int(xLocation + math.sin(angleMajor) * halfMajorLength)
Y1 = int(yLocation - math.cos(angleMajor) * halfMajorLength)
X2 = int(xLocation - math.sin(angleMajor) * halfMajorLength)
Y2 = int(yLocation + math.cos(angleMajor) * halfMajorLength)
cv2.line(Image, (X1, Y1), (X2, Y2), (0, 0, 255), 2)

#Draw minor axis onto video feed and assign colour blue with thickness 2
X3 = int(xLocation + math.sin(angleMinor) * halfMinorLength)
Y3 = int(yLocation - math.cos(angleMinor) * halfMinorLength)
X4 = int(xLocation - math.sin(angleMinor) * halfMinorLength)
Y4 = int(yLocation + math.cos(angleMinor) * halfMinorLength)
cv2.line(Image, (X3, Y3), (X4, Y4), (255, 0, 0), 2)

#_______________________________________________________________________________________________________________________
#Using the centre coordinates of the image frame and traced ellipse to ensure alignment of the robot in the pipe (keeping the ellipse centered)
#This code is to visually represents that and provides necessary readouts for arduino control

h = Image.shape[0] #Find the dimensions of the camera frame
w = Image.shape[1]
halfh = int(h/2) #Half the height and width values to find the coordinates of the centre point of the frame
halfw = int(w/2)
cv2.circle(Image, (halfw, halfh), 2, (255, 255, 255), -1) #Draw a white dot to mark the centre of the frame
cv2.circle(Image, (int(centerCoordinates[0]), int(centerCoordinates[1])), 2, (255, 255, 255), -1) #Draw a white dot to mark the centre of the ellipse
cv2.line(Image, (halfw, halfh), (int(centerCoordinates[0]), int(centerCoordinates[1])), (0, 255, 0), 2) #Draw a line between the two central points

#Computing horizontal and vertical offset in pixels
horizontalOffset = halfw - centerCoordinates[0]
verticalOffset = halfh - centerCoordinates[1]
print('Vertical Offset:', verticalOffset, 'Horizontal Offset:', horizontalOffset)
#In the above due to the coordinate system:
#Horizontal: 0 = Aligned; -ive = Ellipse is right of centre; +ive = Ellipse is left of centre
#Vertical: 0 = Aligned; -ive = Ellipse is lower than centre; +ive = Ellipse is higher than centre

#_______________________________________________________________________________________________________________________
# *** CONDUCTING IN PIPE OBSTACLE DETECTION AND AVOIDANCE USING THE YOLOvX ALGORITHM ***
#_______________________________________________________________________________________________________________________

#Create a mask for the YOLO algorithm to look at that we will project our image feed onto
maskImage = np.zeros(Image.shape, np.uint8) #Creates a black image that us of equal size to our camera feed
cv2.ellipse(maskImage, (xLocation, yLocation), (halfMinorLength, halfMajorLength), angle, 0, 360, (255, 255, 255), -1) #Draws an ellipse on this new image equal to that on the original image
finalImage = cv2.bitwise_and(Image,maskImage) #Puts the mask over the top of the original image keeping the areas of the original image that fall under the white area of the mask but blacking out the rest

#_______________________________________________________________________________________________________________________
#Implementing my YOLO26 trained model onto the new masked image
model1 = YOLO("C:/Users/james/Desktop/PycharmProjects/Dissertation Results/MuSGD/Final end model training completed results/obb/YOLO26_for_machine_vision/weights/best.pt")
results = model1(finalImage) ######, conf = 0.5 may need this inside brackets?
annotated_frame = results[0].plot()

cv2.imshow('Processed Image 2', annotated_frame)
cv2.waitKey(0)

#    for result in results:
#        cv2.imshow('Result', result)
#        for box in i.boxes:
#            x1, y1, x2, y2 = box.xyxy[0]
#            cls = int(box.cls[0])
#            label = model.names[cls]
#    print(f'detected {label} at {x1}, {y1}')


#_______________________________________________________________________________________________________________________
#Show video feed with all items drawn on to visualise robot control:
    #Show the video feed with the ellipse drawn on
cv2.imshow('Processed Image', Image)
cv2.waitKey(0)

cv2.destroyAllWindows()
print("Program Ended")
