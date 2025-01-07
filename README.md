# Im2col SIMD for 2D Tensors

A **SIMD-optimized** implementation of the im2col operation for 2D images (e.g., grayscale). This repository provides C++ source code targeting **AVX2** (x86_64) and **NEON** (Armv8) architectures, as well as a **reference** (non-SIMD) implementation. You can integrate the resulting library into your own C++ project or access it from Python through a simple ctypes wrapper.

![im2col](./img.webp)

## Overview
The **im2col** (“image to column”) operation is commonly used in convolutional neural networks (CNNs) to transform a 2D image (or feature map) into a set of column vectors, facilitating efficient matrix-multiplication-based convolutions.

By taking advantage of **SIMD intrinsics**, we can significantly speed up the im2col computation on CPUs that support vector operations. This repository includes:
- **AVX2** implementation for modern x86_64 processors  
- **NEON** implementation for Armv8 processors  
- **Reference** scalar implementation (no intrinsics) for portability or as a fallback  


