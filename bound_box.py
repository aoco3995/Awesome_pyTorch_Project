import cv2
import numpy as np
import copy

# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
# wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
img = cv2.pyrDown(cv2.imread('pikachu3.jpg', cv2.IMREAD_UNCHANGED))
orig_img = copy.deepcopy(img)
images = []
threshed_img =[]

for i in range(5):
    images.append(copy.deepcopy(img))

i = 1
for image in images:
    # threshold image
    threshed_img.append(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), i*50, 255, cv2.THRESH_BINARY)[1])
    i = i + 1

i = 1
for threshed in threshed_img:
    cv2.imshow("threshhold Image" + str(i), threshed)
    i = i + 1

# find contours and get the external one

for threshed in threshed_img:
    contours, hier = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
    #                cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    print(contours)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if h >= 50:
            crop_img = orig_img[y:y+h, x:x+w]
            cv2.imwrite("crops\\"+str(x)+str(y)+".png", crop_img)
            print(h,w)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    cv2.imshow("contours", img)

    cv2.imshow("contours", img)
    print(type(contours))

while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()