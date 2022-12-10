import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from CNN import CNN

# Set Device
device = torch.device('cpu')
print(device)

in_channels = 3
num_classes = 5
classes = ('pikachu', 'drone', 'dog', 'cat', 'person')

model = CNN(in_channels)
model.to(device)

PATH = '200p_dataset_model_79_percent.pth'
device = torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load(PATH, map_location=device))

def predict_image(image, Threshold):

    """Predict the class of a given image.

    Args:
        image (numpy.ndarray): Image to classify.
        Threshold (float): Threshold for class prediction.

    Returns:
        tuple: A tuple containing the score and predicted class of the input image. If the score is below the threshold, the predicted class will be "none".
    """

    cv2.imshow("",image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200,200))
    #image = np.transpose(image, (1,0,2))
    #print(image)
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(image)
    #print(x.shape)

    x = x.to(device,dtype=torch.float)
    x = x * 255
    scores = model(x)
    #print(scores)
    score_val, prediction = scores.max(1)
    #print(score_val)
    if score_val[0] >= Threshold:
        #print(classes[prediction])
        return score_val[0], str(classes[prediction])
    else:
        return 0,"none"

# while True:
#     key = cv2.waitKey(1)
#     if key == 27: #ESC key to break
#         break