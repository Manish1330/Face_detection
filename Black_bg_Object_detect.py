import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread('img-1.jpg')
img = cv2.resize(img, (800, 800))

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

#printing Count of objects
blur = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blur, 30, 150, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)
 
(cnt, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
 
 
print("Objects in image : ", len(cnt))

# get contours
result = img.copy()
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #printing co-ordinates
    #print("x,y,w,h:", x, y, w, h)

# save resulting image
cv2.imwrite('object_detected.jpg', result)

# show thresh and result
cv2.imshow("bounding_box", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

