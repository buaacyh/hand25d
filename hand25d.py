'''
  author:cuiyunhao
  data:2018.07.10
'''


import torch.nn as nn
import torch

class Hand(nn.Module):
  def __init__(self):
    super(Hand, self).__init__()
    
    self.conv1_1 = nn.Conv2d(3, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # flops: 0.11G
    self.relu = nn.ReLU(inplace = True)
    self.conv1_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # flops:2G
    self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    self.conv2_1 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # 0.6G
    self.conv2_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # 0.15G
    self.conv2_3 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # 0.034G
    self.conv2_4 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # 0 
    self.conv2_5 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1) # 0
    
    self.up = nn.Upsample(scale_factor = 2)
    self.conv3_1 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1)
    self.conv3_2_1 = nn.Conv2d(512, 256, bias = True, kernel_size = 1, stride = 1)
    self.conv3_2_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1)
    self.conv3_3_1 = nn.Conv2d(512, 256, bias = True, kernel_size = 1, stride = 1)
    self.conv3_3_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1)
    self.conv3_4_1 = nn.Conv2d(512, 256, bias = True, kernel_size = 1, stride = 1)
    self.conv3_4_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1)
    self.conv3_5_1 = nn.Conv2d(512, 256, bias = True, kernel_size = 1, stride = 1)
    self.conv3_5_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 3, stride = 1, padding = 1)
    self.conv3_6_1 = nn.Conv2d(512, 512, bias = True, kernel_size = 1, stride = 1)
    self.conv3_6_2 = nn.Conv2d(512, 512, bias = True, kernel_size = 3, stride = 1, padding = 1)

    self.conv4_1 = nn.Conv2d(512, 256, bias = True, kernel_size = 7, stride = 1, padding = 3)
    self.conv4_2 = nn.Conv2d(256, 256, bias = True, kernel_size = 7, stride = 1, padding = 3)
    self.conv4_3 = nn.Conv2d(256, 42, bias = True, kernel_size = 7, stride = 1, padding = 3)

    self.layer1 = nn.Sequential(self.conv1_1, self.relu, self.conv1_2, self.relu, self.maxpool)
    self.layer2_1 = nn.Sequential(self.conv2_1, self.relu, self.maxpool) 
    self.layer2_2 = nn.Sequential(self.conv2_2, self.relu, self.maxpool) 
    self.layer2_3 = nn.Sequential(self.conv2_3, self.relu, self.maxpool) 
    self.layer2_4 = nn.Sequential(self.conv2_4, self.relu, self.maxpool) 
    self.layer2_5 = nn.Sequential(self.conv2_5, self.relu, self.maxpool) 

    self.layer3_1 = nn.Sequential(self.conv3_1, self.relu, self.up)
    self.layer3_2 = nn.Sequential(self.conv3_2_1, self.conv3_2_2, self.up)
    self.layer3_3 = nn.Sequential(self.conv3_3_1, self.conv3_3_2, self.up)
    self.layer3_4 = nn.Sequential(self.conv3_4_1, self.conv3_4_2, self.up)
    self.layer3_5 = nn.Sequential(self.conv3_5_1, self.conv3_5_2, self.up)
    self.layer3_6 = nn.Sequential(self.conv3_6_1, self.conv3_6_2, self.up)


  def forward(self, x):
    x = self.layer1(x)#256@64
    skip1 = x
    x = self.layer2_1(x)#256@32
    skip2_1 = x
    x = self.layer2_2(x)#256@16
    skip2_2 = x
    x = self.layer2_3(x)#256@8
    skip2_3 = x
    x = self.layer2_4(x)#256@4
    skip2_4 = x
    x = self.layer2_5(x)#256@2
    aa = self.layer3_1(x)

    x = torch.cat((self.layer3_1(x), skip2_4), 1)#512@4
    x = torch.cat((self.layer3_2(x), skip2_3), 1)#512@8
    x = torch.cat((self.layer3_3(x), skip2_2), 1)#512@16
    x = torch.cat((self.layer3_4(x), skip2_1), 1)#512@32
    x = torch.cat((self.layer3_5(x), skip1), 1)#512@64
    x = self.layer3_6(x)#512@128

    x = self.conv4_1(x)
    x = self.relu(x)
    x = self.conv4_2(x)
    x = self.relu(x)
    x = self.conv4_3(x)

    H2D = x[:,:21,:,:]
    HZr = x[:,21:,:,:]

    return H2D, HZr


