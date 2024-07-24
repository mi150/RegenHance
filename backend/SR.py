import os
import math
import torch
import cv2
class SRDetector():
    def __init__(self,SRconfig, my_model, ckp):
        self.args = SRconfig

        self.scale = SRconfig.scale

        self.ckp = ckp
        self.model = my_model

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

