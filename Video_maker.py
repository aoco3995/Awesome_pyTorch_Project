#import necessary packages
import os
from glob import glob
from moviepy.editor import *
from tkinter import filedialog
from tkinter import *

#prompt user to select a folder
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory()

#get list of images in folder
imglist = glob(os.path.join(folder_path, '*.jpg'))

#import random module
import random

#select 10 random images from list of images
random.shuffle(imglist)
imgs = imglist[:10]

#create list of clips
clips = [ImageClip(m).set_duration(5)
          for m in imgs]

#concatenate all clips together
concat_clip = concatenate_videoclips(clips, method="compose")

#write video
concat_clip.write_videofile("slideshow.mp4",30)