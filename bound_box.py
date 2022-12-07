import cv2
import numpy as np
import copy
import os
import test_any_size_image_on_cpu
from test_any_size_image_on_cpu import predict_image


def mask_outside_area(img, x1, y1, x2, y2, im_path):
    # create a mask
    mask = np.zeros(img.shape, dtype=np.uint8)

    # define the area
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # apply the mask
    result = cv2.bitwise_and(img, mask)

    # save the result
    #cv2.imwrite(im_path, result)

    return result
    

dir = 'crops'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

#Read image
img = cv2.imread('pickachu_box_tests\pikachu6.jpg', cv2.IMREAD_UNCHANGED)
orig_img = copy.deepcopy(img)
images = []
threshed_img =[]

for i in range(50):
    images.append(copy.deepcopy(img))

i = 1
for image in images:
    # threshold image
    threshed_img.append(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), i*5, 255, cv2.THRESH_BINARY)[1])
    i = i + 1

i = 1
# for threshed in threshed_img:
#     cv2.imshow("threshhold Image" + str(i), threshed)
#     i = i + 1

# find contours and get the external one

for threshed in threshed_img:
    contours, hier = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
    #                cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    #print(contours)
    min_con_area = image.shape[0]*image.shape[1]
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        #print(img.shape)
        if (h > img.shape[0]/8) & (w > img.shape[1]/8):
            #crop_img = orig_img[y:y+h, x:x+w]
            #v2.imwrite("crops\\"+str(x)+str(y)+".png", crop_img)
            if predict_image(mask_outside_area(orig_img,x,y,x+w, y+h, "crops\\"+str(x)+str(y)+".png")) == "pikachu":
                if cv2.contourArea(c) < min_con_area:
                    min_contour = c

#print(h,w)
# draw a green rectangle to visualize the bounding rect
x, y, w, h = cv2.boundingRect(min_contour)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imshow("contours", img)

cv2.imshow("contours", img)
#print(type(contours))

while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()