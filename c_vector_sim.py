import torch
import os
import torch
from a_preprocess import get_embedding

# expects location and returns cosine similarity
def __vector_cmp__(x1,x2,y1,y2,alpha):
    a = os.path.join(x1,x2)
    b = os.path.join(y1,y2)
    a1 = get_embedding(a)
    b1 = get_embedding(b)
    if(torch.cosine_similarity(a1.unsqueeze(0), b1.unsqueeze(0)) >= alpha):
        print(torch.cosine_similarity(a1.unsqueeze(0), b1.unsqueeze(0)))
        return "Duplicate"
    else:
        print(torch.cosine_similarity(a1.unsqueeze(0), b1.unsqueeze(0)))
        return "Not Duplicate"
    # return Preprocess(a)

