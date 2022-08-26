import time

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image

# torch.backends.quantized.engine = 'fbgemm'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

net = models.mobilenet_v2(pretrained=True)
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError('failed to read camera frame')

        # convert to RGB
        image = image[:, [2, 1, 0]]
        permuted = image

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        output = net(input_batch)
        
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # print(probabilities)
        
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
        
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0
