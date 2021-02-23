
import cv2
from torch.autograd import Variable
import torch
from numpy import array
import numpy as np

model = torch.load("/content/temp-000000000-059442.pt")
model.cuda()
model.eval()
image = 1. - cv2.imread("/content/1.png", cv2.IMREAD_GRAYSCALE)/255.
H, W = image.shape
image= image.reshape(1,1,H,W)
print (image)
image = torch.FloatTensor(image)
print ("go")
bimage = model.forward(Variable(image.cuda(), requires_grad=False))
print ("d")
result = array(bimage.data.cpu(), 'f')[0, 0]
print (np.amin(result))
print (np.amax(result))
for i in range(len(result)):
  for j in range(len(result[0])):
    if result[i][j] > 0.17:
      result[i][j] = 255
    else:
      result[i][j] = 0
print (np.amin(result))
print (np.amax(result))
cv2.imwrite("2.jpg",result)
