import cv2
import numpy as np
import copy
import os
import test_any_size_image_on_cpu
from test_any_size_image_on_cpu import predict_image


def mask_outside_area(img, x1, y1, x2, y2):
    """
    Mask an area outside of a given rectangle in an image.
    
    This function creates a black mask with a white rectangle defined by the
    (x1, y1) and (x2, y2) coordinates, and then applies the mask to the input image
    to mask out the area outside of the rectangle.
    
    Args:
        img (numpy.ndarray): the input image as a NumPy array
        x1 (int): the x-coordinate of the top-left corner of the rectangle
        y1 (int): the y-coordinate of the top-left corner of the rectangle
        x2 (int): the x-coordinate of the bottom-right corner of the rectangle
        y2 (int): the y-coordinate of the bottom-right corner of the rectangle
        
    Returns:
        numpy.ndarray: the masked image as a NumPy array
    """


    # create a mask
    mask = np.zeros(img.shape, dtype=np.uint8)

    # define the area
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # apply the mask
    result = cv2.bitwise_and(img, mask)

    # save the result
    #cv2.imwrite(im_path, result)

    return result
    
def get_bound_area(input_img, class_to_look_for, threshold):
    """
    Detect an object of a given class in an image.
    
    This function applies multiple thresholding operations to the input image and
    finds contours in the thresholded images. For each contour that has an area
    greater than 1/8 of the image's total area, it applies a mask to the original
    image to mask out everything outside of the contour and passes the masked 
    image to a `predict_image()` function. If the `predict_image()` function 
    returns a prediction of the class that we are looking for and has a 
    confidence greater than the given `threshold`, the contour with the minimum
    area is saved. Finally, the bounding box coordinates of the minimum contour
    are returned.
    
    Args:
        input_img (numpy.ndarray): the input image as a NumPy array
        class_to_look_for (str): the class of object to look for in the image
        threshold (float): a threshold value for detecting the object
        
    Returns:
        tuple: a tuple containing the bounding box coordinates of the detected
            object (x1, y1, x2, y2)
    """

    # dir = 'crops'
    # for f in os.listdir(dir):
    #     os.remove(os.path.join(dir, f))

    #Read image
    #img = cv2.imread('pickachu_box_tests\pikachu6.jpg', cv2.IMREAD_UNCHANGED)
    cv2.imshow("Input Frame", input_img)
    img = copy.deepcopy(input_img)
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

    min_contour ="not set"
    threshold_to_beat = threshold 
    # find contours and get the external one'
    for threshed in threshed_img:
        contours, hier = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
        #                cv2.CHAIN_APPROX_SIMPLE)

        # with each contour, draw boundingRect in green
        # a minAreaRect in red and
        # a minEnclosingCircle in blue
        #print(contours)
        min_con_area = image.shape[0]*image.shape[1]
        image_area = min_con_area
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            #print(img.shape)
            if (h > img.shape[0]/8) & (w > img.shape[1]/8):
                crop_img = orig_img[y:y+h, x:x+w]
                #v2.imwrite("crops\\"+str(x)+str(y)+".png", crop_img)
                #masked_image = mask_outside_area(orig_img,x,y,x+w, y+h,)
                #cv2.imshow("Frame Under Review", masked_image)
                new_thresh, found_class = predict_image(crop_img, threshold_to_beat)
                if found_class == class_to_look_for: 
                    if new_thresh >= threshold_to_beat:
                        #if cv2.contourArea(c) < min_con_area:
                        min_contour = c

    # w = orig_img.shape[1]/4
    # h = orig_img.shape[0]/4
    # if min_con_area > image_area/4:
    #     for x_pos in range(10):
    #         for y_pos in range(10):
    #             x = orig_img.shape[1]/10*x_pos
    #             y = orig_img.shape[0]/10*y_pos

    #             masked_image = mask_outside_area(orig_img,int(x),int(y),int(x+w),int(y+h))
    #             if predict_image(masked_image, threhold)[1] == class_to_look_for:
    #                 if cv2.contourArea(c) < min_con_area:
    #                     min_contour = c



    #print(h,w)
    # draw a green rectangle to visualize the bounding rect
    if min_contour == "not set":
        x, y, w, h = (0,0,orig_img.shape[1], orig_img.shape[0])
    else:
        x, y, w, h = cv2.boundingRect(min_contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return (x,y), (w,h)

    #print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    cv2.imshow("contours", img)

    cv2.imshow("contours", img)
    #print(type(contours))

# while True:
#     key = cv2.waitKey(1)
#     if key == 27: #ESC key to break
#         break

# cv2.destroyAllWindows()