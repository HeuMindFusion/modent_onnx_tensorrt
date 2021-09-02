"""trt_modnet.py

This script demonstrates how to do real-time "image matting" with
TensorRT optimized MODNet engine.
"""


import argparse

import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.writer import get_video_writer
from utils.background import Background
from utils.display import open_window, show_fps
from utils.display import FpsCalculator, ScreenToggler
from utils.modnet import TrtMODNet


WINDOW_NAME = 'TrtMODNetDemo'



class TrtMODNetRunner():
    

    def __init__(self, modnet):
        self.modnet = modnet

    def run(self):
        """Get img and bg, infer matte, blend and show img, then repeat."""
        
        capture = cv2.VideoCapture(0)
        #capture = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=1280, height=720 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
        while True:
            ret, img = capture.read()
           
            if img is None:  break
            matte = self.modnet.infer(img)
            img_show = img.copy() 
            
            matte_org = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2)
            det_line = (matte * 255).astype('uint8')
            ret,img1=cv2.threshold(det_line, 240, 255, cv2.THRESH_BINARY); 

            masked = cv2.bitwise_and(img, img, mask=img1)
            cv2.imshow(WINDOW_NAME, masked)
            cv2.imshow('img1', img1)
            key = cv2.waitKey(1)
            if key == 27:
                break

    def __del__(self):
        cv2.destroyAllWindows()


def main():
   
    modnet = TrtMODNet()
  
    runner = TrtMODNetRunner(modnet)
    runner.run()

if __name__ == '__main__':
    main()
