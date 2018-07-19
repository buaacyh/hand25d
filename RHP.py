'''
  author:cuiyunhao
  data:2018.07.10
'''

import torch.utils.data as data
import numpy as np
import torch
import scipy.io as sio
import cv2
from utils.img import DrawGaussian

class RHP(data.Dataset):
  def __init__(self, dataset_dir, split):
    print '==> initializing 2D {} data.'.format(split)

    annot = {}
    datadir = '{}/{}/anno_{}.mat'.format(dataset_dir, split, split)
    mat_data = sio.loadmat(datadir)

    for name in mat_data.keys():
      if name.startswith('frame'):
        annot[name] = mat_data[name]

    print 'Loaded 2D {} {} samples'.format(split, len(annot))
    
    self.split = split
    self.dataset_dir = dataset_dir
    self.annot = annot
    self.hmGauss = 1
    self.handjoints = 21
    self.outputimg = 128

  
  def LoadImage(self, index):
    path = '{}/{}/color/{:0>5d}.png'.format(self.dataset_dir, self.split, index)
    img = cv2.imread(path)

    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)

    return img
  
  def GetPartInfo(self, index):
    framename = 'frame{}'.format(index)
    # only left hand
    K = self.annot[framename]['K'][0, 0] # [3,3]
    xy = self.annot[framename]['uv_vis'][0,0][:self.handjoints,:] # [21,3]
    XYZ = self.annot[framename]['xyz'][0,0][:self.handjoints,:] # [21,3]

    xy = xy/320*128
    XYZ = XYZ/320*128
    return K, xy, XYZ
      

  # return: input, target_p, target_z, meta
  # input is the image, [128,128,3]
  # target_p is the 2D Gaussian heatmap H2Dgt, [21, 128, 128]
  # target_z is the Zkrgt(r) * H2Dgt, Zkrgt(r) is Zkrgt to root
  # TODO:s_relative(P_relative to P_3D)
  
  def __getitem__(self, index):
    img = self.LoadImage(index)
    img = img.transpose(2, 0, 1) / 256.
    K, xy, XYZ = self.GetPartInfo(index)

    img = img.astype(np.float32)
    K = K.astype(np.float32)
    XYZ = XYZ.astype(np.float32)

    s = np.linalg.norm(XYZ[0,:] - XYZ[1,:])
    Zrk_relat = np.zeros((self.handjoints,), dtype = np.float64)
    for i in range(self.handjoints):
      Zrk_relat[i] = (XYZ[i,2] - XYZ[0,2])/s

    out1 = np.zeros((self.handjoints, self.outputimg, self.outputimg), dtype=np.float32)
    out2 = np.zeros((self.handjoints, self.outputimg, self.outputimg), dtype=np.float32)

    for i in range(self.handjoints):
      if xy[i,2] != 0:
        #pt = np.rint(xy[i,:2]).astype(np.int32)
        pt = xy[i,:2]
        out1[i] = DrawGaussian(out1[i], pt, self.hmGauss)
        out2[i] = Zrk_relat[i] * out1[i]

    torch_img = torch.from_numpy(img)
    torch_out1 = torch.from_numpy(out1)
    torch_out2 = torch.from_numpy(out2)
    torch_K = torch.from_numpy(K)
    torch_xy = torch.from_numpy(xy)
    torch_XYZ = torch.from_numpy(XYZ)
    return torch_img, torch_out1, torch_out2, torch_K, torch_xy, torch_XYZ
    
  def __len__(self):
    return len(self.annot)

