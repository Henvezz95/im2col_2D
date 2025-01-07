import ctypes
import numpy as np
import os
from pathlib import Path
import platform
from time import time
import cv2

platform_uname = platform.uname().system
parent_dir = os.path.join(os.path.dirname(__file__), os.pardir) 
lib_file = None

if platform_uname == "Windows":
    lib_file = Path(parent_dir) / 'build' / 'Release' / 'im2col.dll'
    if lib_file.is_file() == False:
        lib_file = Path(parent_dir) / 'build' / 'Debug' / 'im2col.dll'
else:
    lib_file = Path(parent_dir) / 'build' / 'libim2col.so'

mydll = ctypes.cdll.LoadLibrary(lib_file)
c_function = mydll.im2col

def im2col_SIMD(img, ker_h, ker_w):
    pad_h = ker_h//2
    pad_w = ker_w//2
    mem_tail = int((8//ker_h*ker_w)+1)
    img_padded = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT)
    H, W = img_padded.shape
    outTensor = np.empty(((H-ker_h+1)*(W-ker_w+1)+mem_tail, ker_h*ker_w), dtype=np.float32)
    height = ctypes.c_short(H)
    width = ctypes.c_short(W)
    rowBlock = ctypes.c_short(ker_h)
    colBlock = ctypes.c_short(ker_w)
    input_img = img_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_tensor = outTensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_function(input_img, width, height, rowBlock, colBlock, output_tensor)
    return outTensor[:-mem_tail]