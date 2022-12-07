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

PATH = './custom_classifier_dataset.pth'
device = torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load(PATH, map_location=device))

def predict_image(image):
    cv2.imshow("",image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200,200))
    image = np.transpose(image, (1,0,2))
    #print(image)
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(image)
    #print(x.shape)

    x = x.to(device,dtype=torch.float)
    x = x * 255
    scores = model(x)
    #print(scores)
    _, prediction = scores.max(1)
    print(classes[prediction])
    return str(classes[prediction])

# while True:
#     key = cv2.waitKey(1)
#     if key == 27: #ESC key to break
#         break