import torch
import torch.nn as nn
import numpy as np
from random import sample

def GMS_train(y, X, S1, hidden_size1, L1, NN_type1, sub_size1, cv_K1, K01, n1, p1, 
double_boot1, cv_on1, stab_sel_on1, lam_schd1, lr_power1, lr01, num_it1, gpu_ind1, save_on1, 
save_file1, save_struc1, batchnorm_on1, verb1, sgd, eta_cand, Loss_func, 
Penalty, eta_on, model, D, penalty_type, w_type):
  gpu_ind = int(gpu_ind1)
  device = torch.device('cpu')
  eta_on = int(eta_on)
  if model == "VC" or model == "nonpara":
    eta_on = 0

  #print(device)
  print("###########################################")
  if gpu_ind == -1:
    device = 'cpu'
    print("Training G via CPU computing starts.")
    print("WARNING: CPU computing would be very slow!")
  else:
    if torch.cuda.is_available():
      device = torch.device('cuda', gpu_ind)
      print("Training G via GPU computing starts!")
      print(device)
    else:
      device = torch.device('cpu')
      print("Training G via CPU computing starts!")
      print("WARNING: CPU computing would be very slow!")
  print("###########################################")
  #C0 = int(C0)
  #torch.set_num_threads(C0)
  #print(device)
  
  if torch.is_tensor(X) == False:
    X = torch.from_numpy(X)
  if torch.is_tensor(y) == False:
    y = torch.from_numpy(y)
  if torch.is_tensor(eta_cand) == False:
    eta_cand = torch.from_numpy(eta_cand)
  if torch.is_tensor(lam_schd1) == False:
    lam_schd1 = torch.from_numpy(lam_schd1)
  lam_schd = lam_schd1.to(device, dtype = torch.float)
  X = X.to(device, dtype = torch.float)
  y = y.to(device, dtype = torch.float)
  eta_cand = eta_cand.to(device, dtype = torch.float)
  #eta_mean =  torch.mean(eta_cand).item()
  
  S = int(S1)
  n = int(n1)
  p = int(p1)
  n_b = int(n/S)
  #ind = range(S*int(n/S)) 
  #print("asfojaosdk")
  #sys.stdout.flush()
  
  int_dat = 1
  if n % S != 0: int_dat = 0
    
  #if X.size()[0] == y.size()[0]:
  #  X = X[ind,:]
  #y = y[ind,:]
  n = y.size(0)
  batchnorm_on = int(batchnorm_on1) 
  sub_size = int(sub_size1)
  K0 = int(K01)
  NN_type = NN_type1
  verb = int(verb1)
  L = int(L1)
  hidden_size = int(hidden_size1)
  K = int(cv_K1)
  stab_sel_on = int(stab_sel_on1)
#  print("OAIJE0a")
#  sys.stdout.flush()
  M = lam_schd.size()[0]
  double_boot = int(double_boot1)
  L1pen_on = int(cv_on1)
  lr_power = lr_power1
  
  lr0 = lr01
  num_it = int(num_it1)
  save_on = int(save_on1)
  save_file = save_file1
  save_struc = save_struc1
  
  if NN_type == "MLP":
    class Net(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p):
        super(Net, self).__init__()
        #input_dim = 2*S + 2
        input_dim = 8*S + 8 + 4
        self.relu = nn.GELU()
        #self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc_out = nn.Linear(hidden_size, p)
        self.layers = nn.ModuleList()
        if batchnorm_on == 1:
          for j in range(L - 1):
            self.layers.append( nn.Linear(hidden_size, hidden_size) )
            self.layers.append( nn.BatchNorm1d(hidden_size) ) 
            self.layers.append( nn.ReLU() ) 
        else:
          for j in range(L - 1):
            self.layers.append( nn.Linear(hidden_size, hidden_size) )
            self.layers.append( nn.ReLU() ) 
 
      def forward(self, w, lam, eta, batchnorm_on):
        out0 = torch.exp(-w)
        out0_lam = torch.exp(-lam)
        out_w = torch.cat([out0/0.2890527, 1.0/torch.sqrt(w + 0.01)/1.256232, torch.sqrt(w + 0.01)/0.4585749, torch.log(w + 0.01)/1.181673, w,  (1.0-out0)/0.2890527, 1.0/(w+1.0)/0.219450, 1.0/(out0+0.01)/8.897686],dim=1)
        out_lam = torch.cat([out0_lam/0.2890527, 1.0/torch.sqrt(lam + 0.01)/1.256232, torch.sqrt(lam + 0.01)/0.4585749, torch.log(lam + 0.01)/1.181673, lam, (1.0-out0_lam)/0.2890527, 1.0/(lam+1.0)/0.219450, 1.0/(out0_lam+0.01)/8.897686],dim=1)
        out_lam *= 10.0
        eta1 = eta#torch.clip(eta, 0.0)
        out_eta = torch.cat([10.0*eta1, torch.log(eta1+0.0001), 10.0 - 10.0*eta1**2, 1.0/torch.sqrt(eta1+0.001)],dim=1)
        out1 = torch.cat([out_w, out_lam, out_eta],dim=1)
        out = self.relu(self.fc0(out1))
        if batchnorm_on == 1:
          for i in range(L-1):
            out = self.layers[3*i](out)
            out = self.layers[3*i+1](out)
            out = self.layers[3*i+2](out)
        else:
          for i in range(L-1):
            out = self.layers[2*i](out)
            out = self.layers[2*i+1](out)
        
        out = self.fc_out(out)
        return out   

  if NN_type == "MLP+Linear":
    class Net(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p):
        super(Net, self).__init__()
        #input_dim = 2*S + 2
        input_dim = 8*S + 8 + 4
        self.relu = nn.GELU()
        #self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc_out = nn.Linear(hidden_size, p)
        self.layers = nn.ModuleList()
        if batchnorm_on == 1:
          for j in range(L - 1):
            self.layers.append( nn.Linear(hidden_size, hidden_size) )
            self.layers.append( nn.BatchNorm1d(hidden_size) ) 
            self.layers.append( nn.ReLU() ) 
        else:
          for j in range(L - 1):
            self.layers.append( nn.Linear(hidden_size, hidden_size) )
            self.layers.append( nn.ReLU() ) 
 
      def forward(self, w, lam, eta, batchnorm_on):
        out0 = torch.exp(-w)
        out0_lam = torch.exp(-lam)
        out_w = torch.cat([out0/0.2890527, 1.0/torch.sqrt(w + 0.01)/1.256232, torch.sqrt(w + 0.01)/0.4585749, torch.log(w + 0.01)/1.181673, w,  (1.0-out0)/0.2890527, 1.0/(w+1.0)/0.219450, 1.0/(out0+0.01)/8.897686],dim=1)
        out_lam = torch.cat([out0_lam/0.2890527, 1.0/torch.sqrt(lam + 0.01)/1.256232, torch.sqrt(lam + 0.01)/0.4585749, torch.log(lam + 0.01)/1.181673, lam, (1.0-out0_lam)/0.2890527, 1.0/(lam+1.0)/0.219450, 1.0/(out0_lam+0.01)/8.897686],dim=1)
        out_lam *= 10.0
        eta1 = eta#torch.clip(eta, 0.0)
        out_eta = torch.cat([10.0*eta1, torch.log(eta1+0.0001), 10.0 - 10.0*eta1**2, 1.0/torch.sqrt(eta1+0.001)],dim=1)
        out1 = torch.cat([out_w, out_lam, out_eta],dim=1)
        out = self.relu(self.fc0(out1))
        if batchnorm_on == 1:
          for i in range(L-1):
            out = self.layers[3*i](out)
            out = self.layers[3*i+1](out)
            out = self.layers[3*i+2](out)
        else:
          for i in range(L-1):
            out = self.layers[2*i](out)
            out = self.layers[2*i+1](out)
        #out_b = self.relu(self.fc_out2(out))
        out = torch.cat([out, out1],dim=1)
        #out = torch.cat([out_a,out_b,out_norm],dim=1)
        out = self.fc_out3(out)
        return out   

  if NN_type == "Linear":
    class Net(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p):
        super(Net, self).__init__()
        #input_dim = 2*S 
        input_dim = 8*S + 8 + 4
        
        self.relu = nn.ReLU()
        self.fb0 = nn.Linear(input_dim, hidden_size)
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn_w = nn.BatchNorm1d(4*S)
        self.bn_norm = nn.BatchNorm1d(S)
        self.fc_out1 = nn.Linear(hidden_size, 8*S)
        self.fc_out2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out3 = nn.Linear(hidden_size+8*S, p)
        self.fc_out4 = nn.Linear(input_dim, p)
        
      def forward(self, w, lam, eta, batchnorm_on):
        out0 = torch.exp(-w)
        out0_lam = torch.exp(-lam)
        out_w = torch.cat([out0/0.2890527, 1.0/torch.sqrt(w + 0.01)/1.256232, torch.sqrt(w + 0.01)/0.4585749, torch.log(w + 0.01)/1.181673, w,  (1.0-out0)/0.2890527, 1.0/(w+1.0)/0.219450, 1.0/(out0+0.01)/8.897686],dim=1)
        out_lam = torch.cat([out0_lam/0.2890527, 1.0/torch.sqrt(lam + 0.01)/1.256232, torch.sqrt(lam + 0.01)/0.4585749, torch.log(lam + 0.01)/1.181673, lam, (1.0-out0_lam)/0.2890527, 1.0/(lam+1.0)/0.219450, 1.0/(out0_lam+0.01)/8.897686],dim=1)
        out_lam *= 10.0
        eta1 = eta#torch.clip(eta, 0.0)
        out_eta = torch.cat([10.0*eta1, torch.log(eta1+0.0001), 10.0 - 10.0*eta1**2, 1.0/torch.sqrt(eta1+0.001)],dim=1)
        out1 = torch.cat([out_w, out_lam, out_eta],dim=1)
        #out1 = self.bn1(out1)
        out000 = self.fc_out4(out1)
        return out000   

  if NN_type == "WM-MLP":
    class Net(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p):
        super(Net, self).__init__()
        #input_dim = 2*S 
        input_dim = 3*S + 6 #8*S + 8 + 4
        self.relu = nn.GELU()
        #self.relu = nn.ReLU()
        self.fb0 = nn.Linear(input_dim, hidden_size)
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn_w = nn.BatchNorm1d(8*S)
        self.bn_lam = nn.BatchNorm1d(5)
        self.bn_norm = nn.BatchNorm1d(S)
        self.bn_out = nn.BatchNorm1d(hidden_size + 5*S)
        self.fc_out0 = nn.Linear(hidden_size, hidden_size)
        self.fc_out1 = nn.Linear(hidden_size, 5*S)
        self.fc_out2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out3 = nn.Linear(8*(2*S ), hidden_size)
        self.fc_out_final = nn.Linear(hidden_size + 5*S, p)
        self.ln_a = nn.LayerNorm( 5*S )
        self.ln_b = nn.LayerNorm( hidden_size )
        self.ln_out = nn.LayerNorm( hidden_size + 5*S)
        #self.fc_out_final = nn.Linear(2*hidden_size+input_dim, p)
        #self.ln0 = nn.LayerNorm( hidden_size )
        self.layers = nn.ModuleList()
        if batchnorm_on == 1:
          for j in range(L - 1):
            self.layers.append( nn.Linear(hidden_size, hidden_size) )
            #self.layers.append( nn.BatchNorm1d(hidden_size) )
            self.layers.append( nn.LayerNorm( hidden_size ) )
            self.layers.append( self.relu ) 
        else:
          for j in range(L - 1):
            self.layers.append( nn.Linear(hidden_size, hidden_size) )
            #self.layers.append( nn.ReLU() ) 
            self.layers.append( self.relu )
            
      def forward(self, w, lam, eta, batchnorm_on):
        out0_w = torch.exp(-w)
        out0_lam = torch.exp(-lam)
        out0_eta = torch.exp(-eta)
        delta = 0.000001
        out1 = torch.cat([out0_w, out0_lam, out0_eta],dim=1)
        out1 = torch.cat([out1, 0.2*torch.log(w+delta),0.2*torch.log(lam+delta),0.2*torch.log(eta+delta), 
        w, lam, eta],dim=1)
        out_w = w.repeat(1,5)
        out1 = self.bn1(out1)
        out = self.relu(self.fc0(out1))
        if batchnorm_on == 1:
          for i in range(L-1):
            out = self.layers[3*i](out)
            out = self.layers[3*i+1](out)
            out = self.layers[3*i+2](out)
        else:
          for i in range(L-1):
            out = self.layers[2*i](out)
            out = self.layers[2*i+1](out)
        out_a = self.fc_out1(self.relu( out) )*out_w 
        out_b = self.fc_out2(out)  
        out = torch.cat([out_a, out_b],dim=1)
        out = self.bn_out(out)
        out = self.fc_out_final(out)
        return out   

#############################################
  if sub_size < S:
    nsub = int(sub_size*n_b)
  else:
    nsub = int(S*n_b)
    
  n1 = float(n)
  G = Net(S, hidden_size, L, batchnorm_on, p).to(device)
  if sgd == 'Adam':
    optimizer = torch.optim.Adam(G.parameters(), lr = lr0)
  sys.stdout.flush()
  if sgd == 'RMSprop':
    lr_U = lr0*2.0
    lr_L = lr0/4.0
    optimizer = torch.optim.RMSprop(G.parameters(), lr= lr0, alpha=0.99, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_L, max_lr=lr_U, step_size_up = 2000)
  if sgd == 'SGD':
    lr_U = lr0*2.0
    lr_L = lr0/4.0
    optimizer = torch.optim.SGD(G.parameters(), lr= lr0, momentum=0.9)
    #optimizer = torch.optim.Adam(G.parameters(), lr=lr0)
#    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_L, max_lr=lr_U, step_size_up = 2000)  
  optimizer.step()
  LOSS = 10000.0*torch.zeros(num_it).to(device)
  alpha = torch.ones(K0,S).to(device)
  A = torch.zeros(n, S)
  for i in range(S):
    ind = range(i*n_b,(i+1)*n_b)
    A[ind,i] = 1
  if int_dat == 0: 
    num_rest = n - n_b*S
    if num_rest > S: print("Warning: S is not appropriate;Please choose another S")
    for i in range(num_rest):
      nnn1 = n_b*S
      A[nnn1+i,i] = 1
    
  A = A.t().to(device) 

  test_size = int(n/K)
  U1 = int(K0/2)
  V = 5
  w1_sample = torch.distributions.exponential.Exponential(torch.ones(U1, sub_size))
  w2_sample = torch.distributions.exponential.Exponential(torch.ones(K0-U1, sub_size))
  w_full_sample = torch.distributions.exponential.Exponential(torch.ones(K0, sub_size))
  w_half_sample = torch.distributions.exponential.Exponential(torch.ones(int(K0/2), sub_size))
  w_one_sample = torch.distributions.exponential.Exponential(torch.ones(V))
  
  beta_half = torch.distributions.Beta(torch.ones(U1,S)/5, torch.ones(U1,S)/5)
  
  loss_min = -10000000.0
  ONE = torch.ones(K0, S).to(device)
  one_size = int(K0*0.2) # 10% is ones
  ind_one = sample(range(one_size, K0), K0-one_size)
  cv_K0_size = int((K0 - one_size)/K)
  if cv_K0_size == 0 and L1pen_on == 1:
    print("K0 is too small to implement a K-fold CV!" )
    print("The generator would NOT be trained appropriately!" )
    
  if L1pen_on == 1:
    for i in ind_one:
      k = sample(range(K),1)
      ind = range(int(k[0]*int(S/K)), int((k[0]+1)*int(S/K)))
      ONE[i,:] = 1.0
      ONE[i,ind] = 0.0
    rand = sample(range(K0),K0) 
    ONE = ONE[rand,:]
    ONE[K0-1,:] = 1.0
  if stab_sel_on == 1:
    for i in range(int(K0/2)):
      ind = sample(range(S), int(S/2))
      ONE[i,:] = 1.0
      ONE[i,ind] = 0.0
    ONE[range(int(K0/2),K0),:] = torch.rand(int(K0/2), S).to(device)

  X0 = X
  y0 = y
  i_sub = range(S)
  ind_sub = torch.ones(n)
  Lam = torch.ones(K0,1).to(device)
  J = 1
  it = 0
  it0 = 0.0
  loss0 = 0.0
  loss0_pen = 0.0
  sys.stdout.flush()
  ONE0 = torch.zeros(K0, S).to(device)
  n_eta = eta_cand.size()[0]
  n1 = float(n)
  eta = 0.5*torch.ones(K0,1).to(device)
  if eta_on == 1:
    ind_eta = np.random.choice(range(n_eta),K0)  
    eta = eta_cand[ind_eta].reshape(K0,1).to(device)
  if L1pen_on == 1 or stab_sel_on == 1:
    ind = np.random.choice(range(M),K0)  
    Lam = torch.exp(torch.log(lam_schd[[ind]]) + 0.2*torch.randn(K0,1).to(device))
  
  alpha = w_full_sample.sample().to(device)
  
  #print(eta)
  #print(eta_on)
  #sys.stdout.flush()
  alpha1 = torch.ones(1,S).to(device)
  while J == 1:
      if eta_on == 1:
        if it % 20 == 5:
          eta += 0.001*torch.randn(K0,1).to(device)
        #ind_eta = np.random.choice(range(n_eta),int(K0/2)) 
        #eta[range(int(K0/2)),:] = eta_cand[ind_eta].reshape(int(K0/2),1).to(device)
        #eta[range(int(K0/2)),:] += 0.05*torch.randn(int(K0/2),1).to(device)
      ##############################
      lr = lr0/((it0+1.0)**lr_power)
      if sgd == 'Adam' or 'SGD':
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
      if sgd == 'RMSprop':
        scheduler.step()
        for param_group in optimizer.param_groups:
           param_group["lr"] = param_group["lr"]/((it0+1.0)**lr_power)
        
      if L1pen_on == 1:
        alpha = w_full_sample.sample().to(device)
        if it % 20 == 0:
          ind_one = sample(range(one_size, K0), K0-one_size)
          ONE0 = torch.zeros(K0, S).to(device)
          for i in range(K0):
            k = sample(range(K),1)
            ind = range(int(k[0]*int(S/K)), int((k[0]+1)*int(S/K)))
            ONE[i,:] = 1.0
            ONE[i,ind] = 0.0
            a = torch.rand(1)
            if a[0] < 0.2:
              k = sample(range(K),2)
              ind = range(int(k[0]*int(S/K)), int((k[0]+1)*int(S/K)))
              ONE[i,:] = 1.0
              ONE[i,ind] = 0.0
              ind = range(int(k[1]*int(S/K)), int((k[1]+1)*int(S/K)))
              ONE[i,ind] = 0.0
          ind = sample(range(K0),int(K0/10))
          ONE[ind,:] = 1.0
        ind = sample(range(K0),int(K0/10))
        alpha[ind,:] = 1.0
        
      if stab_sel_on == 1:
        r = sample(range(S),S)
        ONE = ONE[:,r]
        if it % 20 == 0:
          for i in range(int(K0/2)):
            ind = sample(range(S), int(S/2))
            ONE[i,:] = 1.0
            ONE[i,ind] = 0.0
          ONE[range(int(K0/2),K0),:] = beta_half.sample().to(device)
        alpha = torch.ones(K0,S).to(device)
        stab = 2.0
      else:
        stab = 1.0
      
      if double_boot == 1:
        weight1 = w1_sample.sample().to(device)
        alpha[0:U1, :] = weight1
        weight2 = torch.distributions.gamma.Gamma(weight1, 1.0).sample().to(device)
        alpha[U1:K0, :] = weight2
        m = torch.mean(alpha,1).to(device)
        ONE = torch.ones(K0, S).to(device)
      if double_boot == 0 and stab_sel_on == 0 and L1pen_on == 0:
        alpha = w_full_sample.sample().to(device)

      alpha *= ONE 
      ind_one = sample(range(K0), int(K0/10))
      alpha[ind_one,:] = 1.0
      w1 = torch.matmul(alpha, A).t()
      w0 = w1[ind_sub==1.0,:]

      if L1pen_on == 1 or stab_sel_on == 1:
        ind_lam = np.random.choice(range(M),int(K0)) 
        Lam[range(int(K0)),:] = torch.exp( torch.log(lam_schd[ind_lam].reshape(int(K0),1)) + 0.2*torch.randn(int(K0),1).to(device))
      Theta = G(alpha, Lam, eta, batchnorm_on)
      Theta[Theta != Theta] = 0.0

      loss1 = Loss_func(y0, X0, Theta, eta).to(device)
      loss_log = stab*loss1*w0
      loss_fit = torch.mean(loss_log)  

      if L1pen_on == 0 and stab_sel_on == 0:
        loss = loss_fit
      else: 
        pen = Lam*Penalty(Theta, D).to(device)/K0
        loss_pen = torch.sum(pen)
        loss = (loss_fit  + loss_pen) 
        loss0_pen += loss_pen.item()
  
      optimizer.zero_grad()
      loss.backward() 
      optimizer.step()
      
      LOSS[it] = loss.item()
      loss0 += loss.item()
      
      it0 += 1.0
      it += 1
      if it > (num_it-1):
        J = 0
      if(it+1) % 100==0:
        for param_group in optimizer.param_groups:
          lr1 = param_group["lr"] 
        if stab_sel_on == 0 and double_boot == 0:
            mode = "SingleBoot"
        if stab_sel_on == 1:
            mode = "StabSel"
        else:
            if double_boot == 1:
              mode = "DoubleBoot"
            
            if L1pen_on == 1:
              mode = "CV"
        if verb == 2:
          print('{} [{}/{}], Total loss: {:.4f}, Penalty loss: {:.4f}, lr: {:.7f}, lr_power: {}, device: {}'
              .format(mode, it+1, num_it, loss0/100, loss0_pen/100, lr1, lr_power, device))
          #if NN_type == "lowrank":
          #  print('n: {}, p: {}, subsamp_size: {}, hidden_size: {}, L: {}, K0: {}, S: {},  NN_type: {}, latent_dim: {}'
          #    .format(n, p,  nsub, hidden_size, L, K0, S, NN_type, low))
          #if NN_type == "MLP" or NN_type == "Hadamard":
          print('n: {}, p: {}, subsam_size: {}, hidden_size: {}, L: {}, K0: {}, S: {}, NN_type: {}, Pen_type: {}, Model: {}, SGD_type: {}'
              .format(n, p, nsub, hidden_size, L, K0, S, NN_type, penalty_type, model, sgd))
        if verb == 1:
          percent = float((it+1) * 100)  / num_it
          arrow   = '-' * int(percent/100 * 20 - 1) + '>'
          spaces  = ' ' * (20 - len(arrow))
          print('[%s/%s]'% (it+1, num_it), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent)," Loss: {:.4f}, NN_type: {}, type: {}"
          .format( loss0/100, NN_type, mode), end='\r')
        loss0 = 0.0
        loss0_pen = 0.0
        sys.stdout.flush()
      #if sgd == 'RMSprop':
      #  scheduler.step()
      
  if save_on == 1:
    torch.save(G.state_dict(), save_file)
    file1 = open(save_struc,"w")   
    file1.writelines("NN_type: {}\n".format(NN_type))
    #if NN_type == "lowrank":
    #  file1.writelines("latent dimension: {}\n".format(low))

    file1.writelines("L1pen_on: {}\n".format(L1pen_on))    
    file1.writelines("L: {}\n".format(L))    
    file1.writelines("hidden_size: {}\n".format(hidden_size))    
    file1.writelines("S: {}\n".format(S))    
    file1.close() 
  LOSS = LOSS.cpu().detach().numpy()
  if batchnorm_on == 1:
    bn = "batchnorm on" 
  else:
    bn = "batchnorm off" 
  #eta_cand = eta_cand.cpu().detach().numpy() 
  return G, L1pen_on, double_boot, lam_schd, stab_sel_on, LOSS, K, n, p, S, hidden_size, L, K0, S, NN_type,  bn, gpu_ind#, ONE, Theta, Lam


#Theta = r.fit_GMS[13]
#Lam = r.fit_GMS[14]
