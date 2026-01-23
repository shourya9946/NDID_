# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision import transforms
# from PIL import Image
# import torch
from c_vector_sim import __vector_cmp__


img_folder_path = r"C:\Users\shour\OneDrive\Desktop\test_ukbench"
img_1_name = "ukbench00000.jpg"
img_2_name = "ukbench00001.jpg"
alpha = 0.75
print(__vector_cmp__(img_folder_path,img_1_name,img_folder_path,img_2_name,alpha))
