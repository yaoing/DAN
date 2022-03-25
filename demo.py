import os
import argparse

from PIL import Image
import numpy as np
import cv2

import torch
from torchvision import transforms

from networks.dan import DAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file for evaluation.')
 
    return parser.parse_args()

class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth',
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)
        
        return faces

    def fer(self, path):

        img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        ##  single face detection
        x, y, w, h = faces[0]

        
        img = img0.crop((x,y, x+w, y+h))

        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label

if __name__ == "__main__":
    args = parse_args()

    model = Model()

    image = args.image

    assert os.path.exists(image), "Failed to load image file."

    label = model.fer(image)

    print(f'emotion label: {label}')

