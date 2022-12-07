import cv2
import argparse

def get_frames_from_video(video_file):
  """
  Extract the frames from a video file.
  
  Args:
    video_file: The path to the video file.
  
  Returns:
    A tuple containing the frames per second and a list of frames.
  """
  
  # Open the video file
  video = cv2.VideoCapture(video_file)

  fps = video.get(cv2.CAP_PROP_FPS)

  # Check if the video is opened successfully
  if not video.isOpened():
    raise Exception("Could not open video file!")

  # Create an empty array to store the frames
  frames = []

  # Read the video frame by frame
  while True:
    # Read the next frame
    success, frame = video.read()

    # If there are no more frames to read, break the loop
    if not success:
      break

    # Append the frame to the array
    frames.append(frame)

  # Return the array of frames
  return fps, frames

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

  # Draw the label text on the image
  cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  return image

def create_video_from_images(image_list, output_file, fps):
  """
  Create a video file from a list of images.
  
  Args:
    image_list: A list of images to include in the video.
    output_file: The file name of the output video.
    fps: The frames per second of the output video.
  """
  
  # Get the size of the first image in the list
  # All the images in the list must have the same size
  height, width, channels = image_list[0].shape

  # Define the codec and create the VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

  # Write the frames to the video file
  for image in image_list:
    video.write(image)

  # Release the VideoWriter and destroy all windows
  video.release()

def add_str_to_file_name(file_name: str, string: str) -> str:
  """
  Add a string to the file name before the file extension.
  
  Args:
    file_name: The original file name.
    string: The string to add to the file name.
  
  Returns:
    The modified file name.
  """
  
  # Split the file name into the base name and the extension
  base, ext = file_name.rsplit(".", 1)

  # Add the number before the extension and return the resulting file name
  return f"{base}-{str}.{ext}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="The input video file")
    parser.add_argument("--output", help="Specify an output file")
    args = parser.parse_args()

    input = args.video
    output = add_str_to_file_name(input, "out")

    if args.output is not None:
        output = args.output

    fps, frames = get_frames_from_video(input)
    new_frames = []

    for frame in frames:
        
        label, x, y, height, width, color = get_bounding_box_from_image(frame)
        new_frames.append(draw_rectangle_with_label(frame,"aaa",x,y,height,width,color))
        
    create_video_from_images(new_frames, output, fps)