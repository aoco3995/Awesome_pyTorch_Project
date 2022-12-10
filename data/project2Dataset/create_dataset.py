import os
import cv2

def resize_data(in_dir, img_class, img_class_name, out_dir, out_csv, img_size=(500,500), filetype='.jpg', tt=True, nest=False):

    """
        The resize_data() function resizes images in a directory and writes them to a new directory, along with their labels, in a csv file.

        Arguments
            in_dir: A string representing the path of the directory containing the images to be resized.
            img_class: An integer representing the label of the image class.
            img_class_name: A string representing the name of the image class.
            out_dir: A string representing the path of the directory where the resized images will be saved.
            out_csv: A string representing the path of the csv file where the filename and label of each resized image will be written.
            img_size: A tuple of integers representing the desired size of the resized images.
            filetype: A string representing the filetype of the resized images.
            tt: A boolean flag indicating whether to break the loop after resizing a fixed number of images (4000).
            nest: A boolean flag indicating whether the images are nested in subdirectories within in_dir.
            Returns
            The function does not return anything. It saves the resized images and their labels in out_dir and out_csv, respectively.
    """

    num = 1
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not os.path.exists(out_csv):
        file = open(out_csv, 'w')
        file.write("image,label\n")
        file.close()

    with open(out_csv, 'a') as csvfile:
        # loop through every file in directory
        for file in os.listdir(in_dir):
            
            if ".txt" in file:
                continue

            f = os.path.join(in_dir, file)

            if nest:
                f = os.path.join(f,os.listdir(f)[0])
                #print(f)

            # make sure file is not a directory
            if os.path.isfile(f) and not ".txt" in f and not ".gif" in f:
                # read image
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

                # resize image to img_size and save as file#.jpg
                resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                filename = img_class_name + str(num) + filetype
                filepath = os.path.join(out_dir, filename)
                cv2.imwrite(filepath, resized)
                #print(f"Resized {f}")
                num+=1

                # write image and label to csv
                csvfile.write(filename + "," + str(img_class) + "\n")
                
                if num > 4000:
                    break

        print("Done resizing " + img_class_name)


# dataset params
img_size = (200,200)
out_dir = 'project2Dataset'
classes = {
    "pikachu": 0,
    "drone": 1,
    "dog": 2,
    "cat": 3,
    "person": 4,
}
out_csv = "project2Dataset.csv"


resize_data("cats", classes["cat"], "cat", out_dir, out_csv, img_size=img_size)
resize_data("dogs", classes['dog'], "dog", out_dir, out_csv, img_size=img_size)
resize_data("drones", classes['drone'], "drone", out_dir, out_csv, img_size=img_size)
resize_data("faces", classes['person'], "person", out_dir, out_csv, img_size=img_size, nest=True)
resize_data("pikachu", classes['pikachu'], "pikachu", out_dir, out_csv, img_size=img_size)
