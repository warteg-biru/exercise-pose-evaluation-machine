# Requirements

## Python
- python3.x <= python3.7
    Deep SORT uses tensorflow==1.x which is not supported by python version >= 3.8.  
    You can use pyenv for managing different python versions.  

## GCC and G++
- gcc-8 and g++-8
    gcc-9 and g++9 not supported by CUDA 10

## Graphics Card
- Nvidia Graphics Card 1xxx:
    - CUDA 10 (Should already be installed if ubuntu 3rd party software installation enabled)
    - CUDNN
- AMD GPU
    Refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/prerequisites.md
