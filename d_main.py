import os
from b_p_hash import do_next_or_not
from c_vector_sim import __vector_cmp__
import cv2

img_folder_path = r"C:\Users\shour\OneDrive\Desktop\test_ukbench"
img_1_name = "ukbench00000.jpg"
img_2_name = "ukbench00009.jpg"
alpha = 0.75

im1 = cv2.imread(os.path.join(img_folder_path,img_1_name))
im2 = cv2.imread(os.path.join(img_folder_path,img_2_name))
im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
im1 = cv2.resize(im1, (32,32))
im2 = cv2.resize(im2, (32,32))


if(do_next_or_not(im1,im2)):
    print(__vector_cmp__(img_folder_path,img_1_name,img_folder_path,img_2_name,alpha))
else:
    print("not duplicate_(got_without_triplet_model)")