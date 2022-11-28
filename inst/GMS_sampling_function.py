import torch
import torch.nn as nn
import numpy as np
from random import sample

def GMS_sampling(fit, lam_schd, y, X, B1, B2, B10, gpu_ind1, type1, eta):

##################################
#  N = int(r.N)
  gpu_ind = int(gpu_ind1)
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
  if torch.is_tensor(lam_schd) == False:
    lam_schd = torch.from_numpy(lam_schd)
  lam_schd = lam_schd.to(device, dtype = torch.float)
  M = lam_schd.size()[0]
  sys.stdout.flush()
  eta1 = float(eta)
#G, L1pen_on, double_boot, lam_schd, stab_sel_on, LOSS, K, n, p, S, hidden_size, L, K0, S, NN_type, low#, ONE, Theta, Lam
#  print(X)
#  print(y)
  
      
  G = fit[0].to(device)
  G.train()
  L1pen_on = fit[1] 
  double_boot = fit[2]
  if type1 == "NCV":
    L1pen_on = 2 
  if type1 == "Boot-CV":
    L1pen_on = 1
    double_boot = -1
    
  #lam_schd = fit[3]
  stab_sel_on = fit[4]
  K = fit[6]
  n = fit[7]
  p = fit[8]
  S = fit[9]
  bn = fit[15]
  if bn ==  "batchnorm on":
    batchnorm_on = 1
  else:
    batchnorm_on = 0
#G, L1pen_on, double_boot, lam_schd, stab_sel_on, LOSS, K, n, p, S, hidden_size, L, K0, S, NN_type, low, batchnorm_on#, ONE, Theta, Lam
  n_b = int(n/S)
    
  B1 = int(B1)
  B2 = int(B2)
  B10 = int(B10)

  G.eval()  
  #G = G0.to(device)
###########################################################
  a1_many_sample = torch.distributions.exponential.Exponential(torch.ones(B10,S))
  one2 = torch.ones(B2,B10,S).to(device)
  one3 = torch.ones(B10,S).to(device)
  
  theta_hat = 0.0
  #one3 = torch.ones(B3,S).to(device)
  Theta1 = 0.0
  Theta2 = 0.0
  Theta_cv_one = 0.0
  CV_err = 0.0
  Theta_stab = 0.0
  
  with torch.no_grad():
    #eta = 0.5*torch.ones(B10,1).to(device)
    eta = eta1*torch.ones(B10,1).to(device)
    #print(eta)
    #sys.stdout.flush()
    if type1 == "SingleBoot":
      lam = torch.ones(B10, 1).to(device)
      alpha1 = a1_many_sample.sample().to(device)
      m = torch.mean(alpha1,1).to(device)
      m = m.reshape(B10,1)
      alpha1 = alpha1/m
      Theta1 = G(alpha1, lam, eta, batchnorm_on)
      Theta1 = Theta1.cpu().detach().numpy()
      alpha11 = alpha1.cpu().detach().numpy()
      
      alpha1 = torch.ones(B10, S).to(device)
      theta_hat = G(alpha1, lam, eta, batchnorm_on).cpu().detach().numpy()
      
    if type1 == "DoubleBoot":
      lam = torch.ones(B10,1).to(device)
      alpha1 = a1_many_sample.sample().to(device)
      m = torch.mean(alpha1,1)
      m = m.reshape(B10,1).to(device)
      alpha1 = alpha1/m
      Theta1 = G(alpha1, lam, eta, batchnorm_on)
#      print("ijd9aosdk")
#      sys.stdout.flush()
      alpha1 = alpha1.reshape(1,B10,S).to(device)
      a = torch.ones(B2,B10,S).to(device)*alpha1
      alpha2 = torch.distributions.gamma.Gamma(a, 1.0).sample().to(device)

      m = torch.mean(alpha2,2)
      m = m.reshape(B2,B10,1).to(device)
      alpha2 = alpha2/m
      #alpha2 = torch.distributions.gamma.Gamma(alpha1, 1.0).sample()
      #m = torch.mean(alpha2,2).to(device)
      #alpha2 = alpha2/m.reshape(B2,B10,1)
      lam = torch.ones(B10*B2,1).to(device)
      eta = 0.5*torch.ones(B10*B2,1).to(device)
      Theta2 = G(alpha2.reshape(B2*B10,S), lam, eta, batchnorm_on)
      Theta2 = Theta2.reshape(B2,B10,p)
      alpha11 = alpha1.cpu().detach().numpy()
      
      Theta1 = Theta1.cpu().detach().numpy()
      Theta2 = Theta2.cpu().detach().numpy()
      lam = torch.ones(B10,1).to(device)
      alpha1 = torch.ones(B10, S).to(device)
      eta = 0.5*torch.ones(B10,1).to(device)
      theta_hat = G(alpha1, lam, eta, batchnorm_on).cpu().detach().numpy()
        
      
    w_NN = torch.distributions.exponential.Exponential(torch.ones(B1,S))
    w_lam_NN = torch.distributions.exponential.Exponential(torch.ones(B1,1))
    M = lam_schd.size()[0]
    if type1 == "CV":
#      Theta_cv_boot = torch.zeros(K,M,N,p).to('cpu')
      ind = range(S*int(n/S)) 
      if torch.is_tensor(X) == False:
        X = torch.from_numpy(X)
      if torch.is_tensor(y) == False:
        y = torch.from_numpy(y)
      X = X.to(device, dtype = torch.float)
      y = y.to(device, dtype = torch.float)
      y = y[ind,:]
      X = X[ind,:]
      Theta_cv_one = torch.zeros(K,M,p).to('cpu')
      CV_err = torch.zeros(M).to('cpu')
      #CV_err_boot = torch.zeros(M).to('cpu')
      for k in range(K):
        ind0 = range(int(k*S/K), int((k+1)*S/K))
        ind1 = range(int(k*n/K), int((k+1)*n/K))
        X_test = X[ind1,:]
        y_test = y[ind1,:]
        for j in range(M):
          lam0 = lam_schd[j]#*(1.0-1.0/K)
          lam = torch.ones(1,1).to(device)*lam0
          eta = eta1*torch.ones(1,1).to(device)
          #w = torch.ones(N,n_a).to(device)#w_NN.sample().to(device)
          #w = w_NN.sample().to(device)
          #w[:,ind0] = 0.0
          w_one = torch.ones(1,S).to(device)
          w_one[:,ind0] = 0.0
          Theta_one = G(w_one, lam, eta, batchnorm_on)
          err = Loss_func(y_test, X_test, Theta_one, eta).to(device)
          Theta_cv_one[k,j,:] = Theta_one.reshape(p).cpu()
          #print(err.size())
          #sys.stdout.flush()
          CV_err[j] += torch.mean(err).cpu()/K
        percent = float((k+1) * 100)  / K
        arrow   = '-' * int(percent/100 * 20 - 1) + '>'
        spaces  = ' ' * (20 - len(arrow))
        print('[%s/%s]'% (k+1, K), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
        sys.stdout.flush()
      Theta_cv_one = Theta_cv_one.cpu().detach().numpy() 
      CV_err = CV_err.cpu().detach().numpy() 
      alpha11 = 1.0
      
    if type1 == "Boot-CV": # bootstrapped CV
      ind = range(S*int(n/S)) 
      if torch.is_tensor(X) == False:
        X = torch.from_numpy(X)
      if torch.is_tensor(y) == False:
        y = torch.from_numpy(y)
      X = X.to(device, dtype = torch.float)
      y = y.to(device, dtype = torch.float)
      y = y[ind,:]
      X = X[ind,:]
      
      #w_test = torch.distributions.exponential.Exponential(torch.ones(int(n/K),B10))
      Theta_cv_one = torch.zeros(1).to('cpu')
      CV_err = torch.zeros(M,K,B10).to('cpu')
      A = torch.zeros(n, S)
      
      for i in range(S):
        ind = range(i*n_b,(i+1)*n_b)
        A[ind,i] = 1
      A = A.t().to(device) 
      eta = eta1*torch.ones(B10,1).to(device)
      w_one1 = a1_many_sample.sample().to(device)
      w1 = torch.matmul(w_one1, A).t()
      for k in range(K):
        ind0 = range(int(k*S/K), int((k+1)*S/K))
        ind1 = range(int(k*n/K), int((k+1)*n/K))
        w_one = w_one1.clone()
        w_one[:,ind0] = 0.0
        X_test = X[ind1,:]
        y_test = y[ind1,:]
        #a = torch.rand(int(n/K),B10).to(device)
        #w_test = - torch.log(a)
        w_test = w1[ind1,:]
        for j in range(M):
          lam0 = lam_schd[j]#*(1.0-1.0/K)
          lam = lam0*torch.ones(B10,1).to(device)
          Theta_one = G(w_one, lam, eta, batchnorm_on)
          err = Loss_func(y_test, X_test, Theta_one, eta).to(device)
          #w_test0 = w_test.sample().to(device)
          #print(err.size())
          #sys.stdout.flush()
          #err *= w_test
          CV_err[j,k,:] = err.mean(0).cpu().reshape(B10)
        percent = float((k+1) * 100)  / K
        arrow   = '-' * int(percent/100 * 20 - 1) + '>'
        spaces  = ' ' * (20 - len(arrow))
        print('[%s/%s]'% (k+1, K), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
        sys.stdout.flush()

      Theta_cv_one = Theta_cv_one.cpu().detach().numpy() 
      CV_err = CV_err.cpu().detach().numpy() 
      alpha11 = 1.0

    if type1 == "NCV":
#      Theta_cv_boot = torch.zeros(K,M,N,p).to('cpu')
      ind = range(S*int(n/S)) 
      if torch.is_tensor(X) == False:
        X = torch.from_numpy(X)
      if torch.is_tensor(y) == False:
        y = torch.from_numpy(y)
      X = X.to(device, dtype = torch.float)
      y = y.to(device, dtype = torch.float)
      y = y[ind,:]
      X = X[ind,:]
      Theta_cv_one = torch.zeros(K).to('cpu')
      CV_err = torch.zeros(K).to('cpu')
      for k in range(K):
        ones0 = np.ones(K).reshape(K)
        ones0[k] = 0.0
        ind0 = range(int(k*S/K), int((k+1)*S/K))
        ind1 = range(int(k*n/K), int((k+1)*n/K))
        X_test = X[ind1,:]
        y_test = y[ind1,:]
        idx_k = np.where(ones0 == 1.0)
        CV_err0 = torch.zeros(M).to('cpu')
        #print("oamsd;sdvawoeijg;aij")
        #sys.stdout.flush()
        for h in range(K-1):
          #print("d;sdvawoeijg;aij")
          #sys.stdout.flush()
          u = idx_k[0][h]
          #print("idojqfjg;aij")
          #sys.stdout.flush()
          ind0_in = range(int(u*S/K), int((u+1)*S/K))
          ind1_in = range(int(u*n/K), int((u+1)*n/K))
          X_test_in = X[ind1_in,:]
          y_test_in = y[ind1_in,:]
          eta = eta1*torch.ones(1,1).to(device)
          #print("1092ijdadfawoeijg;aij")
          #sys.stdout.flush()
          for j in range(M):
            lam0 = lam_schd[j]#*(1.0-2.0/K)
            lam = torch.ones(1,1).to(device)*lam0
            w_one = torch.ones(1,S).to(device)
            w_one[:,ind0_in] = 0.0
            w_one[:,ind0] = 0.0
            Theta_one = G(w_one, lam, eta, batchnorm_on)
            err = Loss_func(y_test_in, X_test_in, Theta_one,eta).to(device)
            #Theta_cv_one[k,j,:] = Theta_one.reshape(p).cpu()
            CV_err0[j] += torch.mean(err).cpu()/(K-1)
            #print("fawoeijg;aij")
            #sys.stdout.flush()
          #print("fawoeijg;aij")
        lam1 = lam_schd[ CV_err0 == torch.min(CV_err0) ]
        Theta_cv_one[k] = lam1.item()
        #print(lam1)
        #sys.stdout.flush()
        lam = torch.ones(1,1).to(device)*lam1.item()
        w_one = torch.ones(1,S).to(device)
        w_one[:,ind0] = 0.0
        Theta_one = G(w_one, lam, eta, batchnorm_on)
        err = Loss_func(y_test, X_test, Theta_one,eta).to(device)
        CV_err[k] = torch.mean(err).cpu()
        percent = float((k+1) * 100)  / K
        arrow   = '-' * int(percent/100 * 20 - 1) + '>'
        spaces  = ' ' * (20 - len(arrow))
        print('[%s/%s]'% (k+1, K), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
        sys.stdout.flush()

      #Theta_cv_boot = Theta_boot.cpu().detach().numpy() 
      Theta_cv_one = Theta_cv_one.cpu().detach().numpy() 
      CV_err = CV_err.cpu().detach().numpy() 
      alpha11 = 1.0
      
    
    if stab_sel_on == 1:
      Theta_stab = torch.zeros(M,B10,p).to('cpu')
      ONE = torch.zeros(B10,S).to(device)
      for i in range(B10):
        ONE[i,sample(range(S),int(S/2))] = 1.0 
      for j in range(M):
        lam0 = lam_schd[j]
        lam = torch.ones(B10,1).to(device)*lam0
        Theta_one = G(ONE, lam, eta, batchnorm_on)
        Theta_stab[j,:,:] = Theta_one.cpu()
      Theta_stab = Theta_stab.detach().numpy() 
 
  return  Theta1, Theta2, Theta_cv_one, Theta_stab, CV_err, theta_hat, n, alpha11
