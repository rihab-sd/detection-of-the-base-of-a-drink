import cv2
import numpy as np


#image reading
frame=cv2.imread('62033317.jpg')

height, width = frame.shape[:2]

#Color Orange Parametre
lowerBound_B = np.array([10,100,20])   
upperBound_B = np.array([25,255,255])

#Color White Parametre
lowerBound_G = np.array([0,0,0])
upperBound_G = np.array([179,50,255])


original = frame
kernelOpen = np.ones((5,5))
kernelClosed = np.ones((20,20))


#converting BGR to HSV
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#mask for Orange color
mask_B = cv2.inRange(frame_hsv, lowerBound_B, upperBound_B)
maskOpen_B = cv2.morphologyEx(mask_B,cv2.MORPH_OPEN, kernelOpen)
maskClose_B =cv2.morphologyEx(maskOpen_B, cv2.MORPH_CLOSE, kernelClosed)
maskFinal_B = maskClose_B
contours, hierarchy = cv2.findContours(maskFinal_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for c in contours:
    epsilon = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(original, [approx], -1, (0,0,255), 2)
for c in contours[:]:
    M = cv2.moments(c)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    cv2.circle(original, (cX, cY), 2, (0,0,255), -1)
    cv2.putText(original, "Orange", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#mask for Milk color
mask_G = cv2.inRange(frame_hsv, lowerBound_G, upperBound_G)
maskOpen_G = cv2.morphologyEx(mask_G,cv2.MORPH_OPEN, kernelOpen)
maskClose_G =cv2.morphologyEx(maskOpen_G, cv2.MORPH_CLOSE, kernelClosed)
maskFinal_G = maskClose_G
contours, hierarchy = cv2.findContours(maskFinal_G, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for c in contours:
    epsilon = 0.005 * cv2.arcLength(c, True) #for precision
    approx = cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(original, [approx], -1, (255,50,0), 2)        
for c in contours[:]:
    M = cv2.moments(c)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    cv2.circle(original, (cX, cY), 2, (255,50,0), -1)
    cv2.putText(original, "Milk", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,50,0), 2)

cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('jpg', width, height)
cv2.imshow('jpg', original)
cv2.waitKey(1000000)

cv2.destroyAllWindows()


