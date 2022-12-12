import sys

import cv2 as cv

import torch

from NN_structure import emotion_classifier
from torchvision.transforms import transforms
import cv2 as cv


import pickle


def main():

    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']
    device = 'cuda'
    print("Loading device")
    red = emotion_classifier(out=7).to(device=device)
    checkpoint = torch.load('.\modeloRed.pkl', map_location=device)
    red.load_state_dict(checkpoint)
    red.eval()
    print("Loaded model on CUDA")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image_route = sys.argv[1]

    image = cv.imread(image_route)
    original_image = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    print("predicting")
    with torch.no_grad():
        ouput = red(image.to('cuda'))
    output_label = torch.topk(ouput, 1)
    pred_class = labels[int(output_label.indices)]
    print(pred_class)
    print("Return")


if __name__ == "__main__":
    main()
