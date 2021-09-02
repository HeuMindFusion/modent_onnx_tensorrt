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
from utils.mobileNet import TrtMobileNet


WINDOW_NAME = 'TrtMODNetDemo'



class TrtMODNetRunner():
    

    def __init__(self, mobileNet):
        self.mobileNet = mobileNet

    def run(self):
        """Get img and bg, infer matte, blend and show img, then repeat."""   
        img = cv2.imread('./dog.jpg')
        
        out = self.mobileNet.infer(img)
        print(out.shape)
        print(np.where(out==np.max(out)))
        print(np.max(out))
            

    def __del__(self):
        cv2.destroyAllWindows()


def main():
   
    modnet = TrtMobileNet()
  
    runner = TrtMODNetRunner(modnet)
    runner.run()

if __name__ == '__main__':
    main()
