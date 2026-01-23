import torch
import os
import torch
from a_preprocess import return_embedding , preprocess_image

# expects location and returns cosine similarity
def __vector_cmp__(x1,x2,y1,y2,alpha):
    a = preprocess_image(os.path.join(x1,x2))
    b = preprocess_image(os.path.join(y1,y2))
    a1 = return_embedding(a)
    b1 = return_embedding(b)
    if(torch.cosine_similarity(a1, b1) >= alpha):
        print(torch.cosine_similarity(a1, b1))
        return "Duplicate"
    else:
        print(torch.cosine_similarity(a1, b1))
        return "Not Duplicate"
    # return Preprocess(a)

