import argparse

from PIL import Image
from typing import Optional, Tuple
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
import numpy as np
import cv2
import os.path as osp
import torch.nn.functional as F

import torch
from torchvision import transforms
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from networks.dan import DAN
from rich.console import Console

CONSOLE = Console()

def parse_args():
    parser = argparse.ArgumentParser(prog='video based emotion recognition')
    parser.add_argument('--video', type=str, default=None, help='path to video.')
    parser.add_argument('--camera', action='store_true', help='use camera.')
    parser.add_argument('--gradcam', action='store_true', help='use gradcam.')
    parser.add_argument('--target-layer', type=str, default='head_1', help='gradcam target layer.')
    parser.add_argument('--modalities',
                        type=str,
                        default=['video'],
                        nargs='+',
                        choices=['video', 'audio'],
                        help='modalities to use.')
    parser.add_argument('--model',
                        type=str,
                        default='./models/affecnet8_epoch5_acc0.6209.pth',
                        help='path to model.')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use.')
    parser.add_argument('--fps',
                        type=int,
                        default=30,
                        help='fps of output video.')
    return parser.parse_args()

class ModelInference():
    def __init__(self, args):
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        self.device = torch.device(args.device)
        self.test_pipeline = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        self.gradcam_target = {
            'head_0': self.model.cat_head0.sa,
            'head_1': self.model.cat_head1.sa,
            'head_2': self.model.cat_head2.sa,
            'head_3': self.model.cat_head3.sa
        }
        checkpoint = torch.load(args.model,
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        # TODO: find better face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def numpy_to_pil(self, img: np.array) -> Image.Image:
        """Convert numpy array to PIL format.

        Args:
            img (np.array): input image.

        Returns:
            Image.Image: output image.
        """
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            # print(f'Failed to convert to PIL: {e}')
            return None
        pil_img = Image.fromarray(img_rgb)
        return pil_img

    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in image."""
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            # print(f'Failed to detect faces: {e}')
            return []
        faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        return faces

    def fer(self, np_frame: np.ndarray) -> np.ndarray:
        """Perform facial expression recognition on a frame.

        Visualize the result on the frame.

        Args:
            frame (np.array): input frame

        Returns:
            np.ndarray: returns frame with visualization.
        """
        faces = self.detect_faces(np_frame)
        frame = self.numpy_to_pil(np_frame)
        for (x, y, w, h) in faces:
            cropped = frame.crop((x,y, x+w, y+h))
            # cropped = frame[y:y + h, x:x + w]
            cropped = self.test_pipeline(cropped)
            cropped = cropped.view(1, 3, 224, 224)
            cropped = cropped.to(self.device)

            with torch.set_grad_enabled(False):
                out, _, _ = self.model(cropped)
                out = F.softmax(out[0], dim=0)
                conf, pred = torch.max(out, dim=0)
                index = int(pred)
                label = self.labels[index]

            cv2.rectangle(np_frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            cv2.putText(np_frame, label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return np_frame

class ModelInferenceGradCam(torch.nn.Module):
    """Wrapper for DAN models such that only the prediction is returned.
    By default DAN models return a tupe (x, h) and not only the prediction.

    Args:
        torch (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def get_frame(src: str, frame_id: int) -> Optional[np.ndarray]:
    """Get a single frame from a video file path.

    Args:
        cam (cv2.VideoCapture): video capture object.
        frame_id (int): frame id to load.

    Raises:
        VideoProcessingError: raise if frame id is out of bounds.

    Returns:
        np.ndarray: loaded frame.
    """
    cam = cv2.VideoCapture(src)
    try:
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_id >= 0 & frame_id <= total_frames:
            cam.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cam.read()
            cam.release()
            if ret:
                return frame
    finally:
        cam.release()


def get_total_frames(src: str) -> int:
    """Get the total frame count of a video file.

    Args:
        src (str): path to video.

    Returns:
        int: frame count of video.
    """
    cam = cv2.VideoCapture(src)
    total_frames: int = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    cam.release()
    return total_frames


def get_fps(src: str) -> int:
    """Get fps of a video file.

    Args:
        src (str): path to video.

    Returns:
        int: fps of video.
    """
    cam = cv2.VideoCapture(src)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    return fps


def get_video_frame_size(src: str) -> Tuple[int, int]:
    """Get the frame size of a video file.

    Args:
        src (str): path to video.

    Returns:
        Tuple[int, int]: frame size of video.
    """
    cam = cv2.VideoCapture(src)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam.release()
    return width, height


def main():
    args = parse_args()
    model = ModelInference(args)

    if args.gradcam:
        model_wrapper = ModelInferenceGradCam(model.model)
        gradcam_model = GradCAMPlusPlus(model_wrapper,
                                        target_layer=model.gradcam_target[args.target_layer],
                                        use_cuda=True)
        gradcam_model.batch_size = 32

    if args.video:
        vid_fps = get_fps(args.video)
        if vid_fps == 0:
            vid_fps = 25
        out_file = osp.splitext(args.video)[0] + '_demo.mp4'
        out_vid = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                  vid_fps, get_video_frame_size(args.video))
        frame_count = get_total_frames(args.video)
        duration = int(frame_count / vid_fps)
        if args.fps > vid_fps:
            args.fps = vid_fps

        frame_ids = np.linspace(0, frame_count, num=int(duration * args.fps), dtype=int)
        for i in tqdm(range(frame_count)):
            frame = get_frame(args.video, i)
            if i in frame_ids:
                frame = model.fer(frame)
            out_vid.write(frame)

        out_vid.release()
        CONSOLE.print(f'Stored output video at: {out_file}', style='green')

    if args.camera:
        cap = cv2.VideoCapture(0)
        if args.gradcam:
            out_grad_f = 'gradcam.mp4'
            cam_vid = cv2.VideoWriter(out_grad_f,
                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                            int(cap.get(cv2.CAP_PROP_FPS)),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            CONSOLE.print(f'Storing gradcam video at: {out_grad_f}',
                          style='green')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_frame = model.fer(frame)  # ~ 0.04s inference on GPU
            cv2.imshow('frame', out_frame)

            if args.gradcam:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = np.float32(frame_rgb) / 255
                input_tensor = preprocess_image(frame_rgb,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

                gradcam_output = gradcam_model(input_tensor=input_tensor,
                                               target_category=None,
                                               aug_smooth=False,
                                               eigen_smooth=False,)

                gradcam_output = gradcam_output[0, :]
                frame_gradcam = show_cam_on_image(frame_rgb, gradcam_output, use_rgb=True)
                frame_gradcam = cv2.cvtColor(frame_gradcam, cv2.COLOR_RGB2BGR)
                cam_vid.write(frame_gradcam)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if args.gradcam:
            cam_vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
