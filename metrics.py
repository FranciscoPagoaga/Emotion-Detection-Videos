import sys

import cv2 as cv
import numpy as np
import torch

from NN_structure import emotion_classifier
from torchvision.transforms import transforms
import cv2 as cv


import pickle

def visualize(input, faces, fps, emotion, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.putText(input, emotion,(coords[0]-1,coords[1]-2) ,cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
            # print(face1_feature)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    detector = cv.FaceDetectorYN.create(
        'face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']
    device = 'cuda'
    print("Loading device")
    red = emotion_classifier(out=7).to(device=device)
    checkpoint = torch.load('.\modeloRed_2.pkl', map_location=device)
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

    # image_route = sys.argv[1]

    # image = cv.imread(image_route)
    # original_image = image.copy()
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # image = transform(image)
    # image = torch.unsqueeze(image, 0)
    # print("predicting")
    # with torch.no_grad():
    #     ouput = red(image.to('cuda'))
    # output_label = torch.topk(ouput, 1)
    # pred_class = labels[int(output_label.indices)]
    # print(pred_class)
    # print("Return")


    tm = cv.TickMeter()
    print(len(sys.argv))
    if(len(sys.argv) > 1):
        cap = cv.VideoCapture(sys.argv[1])
    else:
        cap = cv.VideoCapture(0)
            
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*1.0)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*1.0)
    detector.setInputSize([frameWidth, frameHeight])

    while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv.resize(frame, (frameWidth, frameHeight))


            emotionFrame = cv.resize(frame, (frameWidth, frameHeight))
            emotionFrame = cv.cvtColor(emotionFrame, cv.COLOR_BGR2RGB)
            emotionFrame = transform(emotionFrame)
            emotionFrame = torch.unsqueeze(emotionFrame, 0)

            with torch.no_grad():
                ouput = red(emotionFrame.to('cuda'))
            output_label = torch.topk(ouput, 1)
            pred_class = labels[int(output_label.indices)]

            # Inference
            tm.start()
            faces = detector.detect(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS(), pred_class)

            # Visualize results
            cv.imshow('Live', frame)
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
