import torch
import torch.nn as nn
import numpy as np
from random import sample

def GMS_generator(fit, w, lam, eta):

##################################
#  N = int(r.N)
  gpu_ind = fit[16]
  device = torch.device('cpu')
#  print("###########################################")
  if torch.cuda.is_available():
    device = torch.device('cuda', gpu_ind)
#    print("Training G via GPU computing starts!")
#    print(device)
  else:
    device = torch.device('cpu')
#    print("Training G via CPU computing starts!")
#    print("WARNING: CPU computing is not efficient in training the generator.")
#  print("###########################################")
  sys.stdout.flush()
  if torch.is_tensor(w) == False:
    w = torch.from_numpy(w)
  w = w.to(device, dtype = torch.float)
  if torch.is_tensor(lam) == False:
    lam = torch.from_numpy(lam)
  lam = lam.to(device, dtype = torch.float)
  if torch.is_tensor(eta) == False:
    eta = torch.from_numpy(eta)
  eta = eta.to(device, dtype = torch.float)
  
  B = w.size(0)
  #lam1 = lam*torch.ones(B,1).to(device)
  #if eta == 0.5:
  #eta1 = eta*torch.ones(B,1).to(device)
  #else:
  #  eta1 = eta*torch.ones(B,1).to(device)

  
  G = fit[0].to(device)
  G.eval()
  p = fit[8]
  S = fit[9]
  bn = fit[15]
  if bn ==  "batchnorm on":
    batchnorm_on = 1
  else:
    batchnorm_on = 0
#G, L1pen_on, double_boot, lam_schd, stab_sel_on, LOSS, K, n, p, S, hidden_size, L, K0, S, NN_type, low, batchnorm_on#, ONE, Theta, Lam
  #n_b = int(n/S)
    
  #G = G0.to(device)
###########################################################
  
  with torch.no_grad():
    Theta = G(w, lam, eta, batchnorm_on).cpu().detach().numpy() 
  return  Theta





