#import the necessary packages 
import cv2 
import bound_box
from bound_box import get_bound_area
from test_any_size_image_on_cpu import predict_image
import numpy as np

def array_to_mp4(image_array, video_name, fps=30):

    """Converts an array of images to an MP4 video.

    Args:
        image_array (numpy.ndarray): Array of images to convert.
        video_name (str): Name of the output video file.
        fps (int, optional): Frame rate of the output video. Defaults to 30.
    """
    
    # get the size of the array
    height, width, layers = np.shape(image_array[0])
    
    # define the video codec
    video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    # create the video writer
    video = cv2.VideoWriter(video_name, video_codec, fps, (width,height))
    
    # add each image to the video
    for image in image_array:
        video.write(image)
        
    # release the video writer
    video.release()

def draw_rectangle_with_label(image, label, x, y, width, height, color):
  """
  Draw a rectangle on an image with a given label.

  Args:
    image: The image on which to draw the rectangle.
    label: The label to display on the image.
    x: The x coordinate of the rectangle.
    y: The y coordinate of the rectangle.
    width: The width of the rectangle.
    height: The height of the rectangle.
    color: The color of the rectangle (as an (R, G, B) tuple).

  Returns:
    The image with the rectangle and label drawn on it.
  """

  # Draw the rectangle on the image
  cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

  # Calculate the size of the label text
  text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

  # Calculate the coordinates of the label text
  text_x = x + 2
  text_y = y + text_size[1] + 2

  # Draw a black background before the text
  cv2.rectangle(image, (text_x, y), (text_x+text_size[0], text_y+text_size[1]*2), (0,0,0), -1)
  # Draw the label text on the image
  cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  return image

image_array = []
#load the video 
video = cv2.VideoCapture('Test video\Test_Vid.mp4') 

#create a while loop to read the video frames 
i = 61
images_drawn = 0
first_region = False
while True: 
    #read the next frame from the video 
    (grabbed, frame) = video.read() 
    # cv2.imshow("Fram", frame)
    # cv2.imwrite("frame.jpg", frame)
  
    #if the video has ended, then break out of the loop 
    if not grabbed: 
        break 
    
    if i > 30:
        Threshold, bounding_class = predict_image(frame, -1)
        box_region = get_bound_area(frame, bounding_class, Threshold)
        i = 0
        first_region = True
        images_drawn = images_drawn + 1
    i = i + 1
    #draw a bounding box around the frame 
    if first_region == True:
        #cv2.rectangle(frame, box_region[0], box_region[1], (255, 0, 0), 2) 
        image_array.append(draw_rectangle_with_label(frame, bounding_class, box_region[0][0], box_region[0][1], box_region[1][0], box_region[1][1], (255,0,0)))
  
    #show the frame 
    cv2.imshow("Frame", frame) 
  
    #wait for the user to press a key 
    key = cv2.waitKey(1) 
    
    if images_drawn > 60:
        break
    #if the user presses 'q', then break out of the loop 
    if key == ord("q"): 
        break 
  
array_to_mp4(image_array, "Demo.mp4", fps=30)

#release the video capture object 
video.release() 

#close all windows 
cv2.destroyAllWindows()