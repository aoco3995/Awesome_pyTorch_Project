import os
import cv2

def resize_data(in_dir, img_class, img_class_name, out_dir, out_csv, img_size=(500,500), filetype='.jpg', tt=True, nest=False, num = 1, max=4000):
    start = num
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
            if os.path.isfile(f) and not ".txt" in f and not ".gif" in f and not ".py" in f:
                # read image
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                # resize image to img_size and save as file#.jpg
                #print(img_class_name + str(num))
                resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                filename = img_class_name + str(num) + filetype
                filepath = os.path.join(out_dir, filename)
                cv2.imwrite(filepath, resized)
                #print(f"Resized {f}")
                num+=1

                # write image and label to csv
                csvfile.write(filename + "," + str(img_class) + "\n")
                
                if num - start > max:
                    break

        print(f"Done resizing {in_dir} at {img_class_name} {num}")
        return num


# dataset params
img_size = (200,200)
out_dir = '200p_dataset'
classes = {
    "pikachu": 0,
    "drone": 1,
    "dog": 2,
    "cat": 3,
    "person": 4,
}
out_csv = out_dir+".csv"

if os.path.exists(out_csv):
    file = open(out_csv, 'w')
    file.write("image,label\n")
    file.close()

cat_index       = resize_data("cats",         classes["cat"],     "cat",      out_dir, out_csv, img_size=img_size)
dog_index       = resize_data("dogs",         classes['dog'],     "dog",      out_dir, out_csv, img_size=img_size)
drone_index     = resize_data("drones",       classes['drone'],   "drone",    out_dir, out_csv, img_size=img_size)
drone_index     = resize_data("more_drones",  classes['drone'],   "drone",    out_dir, out_csv, img_size=img_size, num=drone_index)
person_index    = resize_data("faces",        classes['person'],  "person",   out_dir, out_csv, img_size=img_size, nest=True, max=500)
person_index    = resize_data("pedestrian0",  classes['person'],  "person",   out_dir, out_csv, img_size=img_size, num=person_index)
person_index    = resize_data("pedestrian1",  classes['person'],  "person",   out_dir, out_csv, img_size=img_size, num=person_index)
person_index    = resize_data("human",        classes['person'],  "person",   out_dir, out_csv, img_size=img_size, num=person_index)
pikachu_index   = resize_data("pikachu",      classes['pikachu'], "pikachu",  out_dir, out_csv, img_size=img_size)
