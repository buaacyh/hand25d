"""
    author: cuiyunhao
"""

import torch
from scipy.optimize import leastsq

def getPreds2D(hm):
  assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
  res = hm.shape[2]
  hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
  idx = torch.argmax(hm, dim = 2)
  preds = torch.zeros((hm.shape[0], hm.shape[1], 2))
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res
  
  return preds

def getPredsZkr(HZr, xy2D):
  # HZr: batch*21*320*320
  # xy2D: batch*21*2
  assert len(HZr.shape) == 4, 'HZr must be a 4-D tensor'
  assert len(xy2D.shape) == 3, 'xy2D must be a 3-D tensor'
  Zkr_s = torch.zeros((HZr.shape[0], HZr.shape[1]))
  for i in range(HZr.shape[0]):
    for j in range(HZr.shape[1]):
      Zkr_s[i, j] = HZr[i, j, xy2D[i, j, 0].to(torch.int32), xy2D[i, j, 1].to(torch.int32)]
  return Zkr_s

def getPredsZroot(xy, XYZ):
  assert len(xy.shape) == 3, 'xy2D must be a 3-D tensor'
  assert len(XYZ.shape) == 3, 'Zkr_s must be a 3-D tensor'
  # n is the joints 1, m is the joints root 0
  xn, xm = xy[:, 1, 0], xy[:, 0, 0]
  yn, ym = xy[:, 1, 1], xy[:, 0, 1]
  Znr, Zmr = Zkr_s[:, 1], Zkr_s[:, 2]
  a = (xn - xm)**2 + (yn - ym)**2
  b = Znr*(xn**2 + yn**2 - xn*xm - yn*ym) + Zmr*(xm**2 + ym**2 - xn*xm - yn*ym)
  c = (xn*Znr - xm*Zmr)**2 + (yn*Znr - ym*Zmr)**2 + (Znr - Zmr)**2 - 1
  #print("a:{}".format(a))
  #print("b:{}".format(b))
  #print("c:{}".format(c))
  Zroot_s = (torch.sqrt(b**2 - 4*a*c) - b)/(2*a)
  #print("Zroot_s:{}".format(Zroot_s))
  #exit(1)
  return Zroot_s

def leastsq_s(XYZ_s):
  joints_connect = [[0,4], [4,3], [3,2], [2,1],
                    [0,8], [8,7], [7,6], [6,5],
                    [0,12], [12,11], [11,10], [10,9],
                    [0,16], [16,15], [15,14], [14,13],
                    [0,20], [20,19], [19,18], [18,17]]
  x = torch.zeros((XYZ_s.shape[0], 20)) # batch*20
  for num in range(20):
    x[:,i] = torch.sqrt(XYZ_s[:, joints_connect[i][0], :], XYZ_s[:, joints_connect[i][1], :])
  y = distance_joints = [0.0461266, 0.04512048, 0.03883177, 0.03410153, 0.10972843, 0.03997845,
  0.03147184, 0.0311841,  0.10868313, 0.04027495, 0.03330259, 0.03451282,
  0.10341862, 0.03536041, 0.0310081,  0.03445587, 0.09839079, 0.03839774,
  0.02125731, 0.02470566]

  sumxy = torch.sum(x*y, 1)
  sumxx = torch.sum(x**2, 1)
  s = sumxy/sumxx
  return s
