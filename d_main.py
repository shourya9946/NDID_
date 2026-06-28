import os
from b_p_hash import do_next_or_not
from c_vector_sim import __vector_cmp__
import cv2

img_folder_path = r"C:\Users\shour\Downloads"
# img_folder_path2 = r"C:\Users\shour\Downloads\archive\train\wingsuit flying"
img_1_name = "WhatsApp Image 2026-05-22 at 9.23.17 PM.jpeg"
img_2_name = "WhatsApp Image 2026-05-22 at 9.23.17 PM (1).jpeg"
alpha = 0.9

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