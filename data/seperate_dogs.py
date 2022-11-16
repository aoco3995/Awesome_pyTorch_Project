import os

for file in os.listdir("dogs_cats_train"):
    fold = os.path.join("dogs_cats_train",file)
    if "dog" in file:
        fnew = os.path.join("dogs",file)
        os.rename(fold, fnew)
    if "cat" in file:
        fnew = os.path.join("cats",file)
        os.rename(fold, fnew)