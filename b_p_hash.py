import numpy as np
import math
c1 = 1/np.sqrt(2)
c2 = 1
def cu(a):
    if a == 0:
        return c1
    else:
        return c2
def DCT(im):
    matrix = np.zeros((32,32))
    
    for u in range(32):
        for v in range(32):
            a = 0.0
            for x in range(32):
                for y in range(32):
                    a += (im[x][y]* math.cos(((2*x + 1) * u * math.pi) / (64)) * math.cos(((2*y + 1) * v * math.pi) / (64)))
            a *= 0.25*cu(u)*cu(v)        
            matrix[u,v] = a
    return matrix

def hash_find(im):
    mat1 = DCT(im)
    h = mat1[0:8,0:8]
    avg = np.sum(h)/64
    k = h.reshape(64).copy()
    # print(k)
    # print(avg)
    r = np.zeros(64)
    for i in range(64):
        if(k[i] >= avg):
            r[i] = 1
        else:
            r[i] = 0
    # print(r)
    return r

def do_next_or_not(im1,im2):
    a1 = hash_find(im1)
    a2 = hash_find(im2)
    if (64 - np.sum((a1 == a2)))>30:
        return 0
    else:
        return 1
