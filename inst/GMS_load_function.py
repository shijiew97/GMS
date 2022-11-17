import torch
import torch.nn as nn
import numpy as np
from random import sample

def GMS_load(Option, Save_file, gpu_ind1, S1, n1, p1, batchnorm_on1,
sub_size1, K01, NN_type1, verb1, L1, hidden_size1, cv_K1, stab_sel_on1, cv_on1,
double_boot1, lr_power1, lr01, num_it1, lam_schd1, y, X, B1, B2, B10, type1, 
eta, eta_on1, model1, eta_cand1, Loss_func, Penalty, penalty_type, sgd1, w1, lam1, eta11):
  
  gpu_ind = int(gpu_ind1)
  if gpu_ind == -1:
    print("WARNING: CPU computing would be very slow!")
  else:
    if torch.cuda.is_available():
      device = torch.device('cuda', gpu_ind)
    else:
      device = torch.device('cpu')
      print("WARNING: CPU computing would be very slow!")
    
  eta_on = int(eta_on1)
  model = str(model1)
  if model == "VC" or model == "nonpara":
    eta_on = 0
  
  lam_schd = lam_schd1
  if torch.is_tensor(lam_schd) == False:
    lam_schd = torch.from_numpy(lam_schd)
  lam_schd = lam_schd.to(device, dtype = torch.float)
  M = lam_schd.size()[0]
  sys.stdout.flush()
  eta1 = float(eta)
  
  if torch.is_tensor(X) == False:
    X = torch.from_numpy(X)
  if torch.is_tensor(y) == False:
    y = torch.from_numpy(y)
  
  eta_cand = np.array(eta_cand1)
  if torch.is_tensor(eta_cand) == False:
    eta_cand = torch.from_numpy(eta_cand)
  if torch.is_tensor(lam_schd1) == False:
    lam_schd1 = torch.from_numpy(lam_schd1)
    lam_schd = lam_schd1.to(device, dtype = torch.float)
  X = X.to(device, dtype = torch.float)
  y = y.to(device, dtype = torch.float)
  eta_cand = eta_cand.to(device, dtype = torch.float)

  
  S = int(S1)
  n = int(n1)
  p = int(p1)
  n_b = int(n/S)
  ind = range(S*int(n/S)) 
  sgd = str(sgd1)
  
  if X.size()[0] == y.size()[0]:
    X = X[ind,:]
  y = y[ind,:]
  n = y.size(0)
  batchnorm_on = int(batchnorm_on1) 
  sub_size = int(sub_size1)
  K0 = int(K01)
  NN_type = str(NN_type1)
  verb = int(verb1)
  L = int(L1)
  hidden_size = int(hidden_size1)
  K = int(cv_K1)
  stab_sel_on = int(stab_sel_on1)
  M = lam_schd.size()[0]
  double_boot = int(double_boot1)
  L1pen_on = int(cv_on1)
  lr_power = lr_power1
  lr0 = lr01
  num_it = int(num_it1)

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

  if NN_type == "simple-MLP":
    class Net(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(S, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, p)
      def forward(self, w, lam, eta, batchnorm_on):
        out = torch.exp(-w)
        out = self.relu( self.fc1(out) )
        out = self.relu( self.fc2(out) )
        out = self.fc_out(out)
        return out   
      
  if sub_size < S: nsub = int(sub_size*n_b)
  else: nsub = n
  n1 = float(n)
  G = Net(S, hidden_size, L, batchnorm_on, p).to(device)
  G.load_state_dict(torch.load(Save_file))
  #print("Successfully loading trained Generator!")
  
  if Option == "Generator":
    G.eval()
    
    if torch.is_tensor(w1) == False:
      w1 = torch.from_numpy(w1)
    w1 = w1.to(device, dtype = torch.float)
    if torch.is_tensor(lam1) == False:
      lam1 = torch.from_numpy(lam1)
    lam1 = lam1.to(device, dtype = torch.float)
    if torch.is_tensor(eta11) == False:
      eta11 = torch.from_numpy(eta11)
    eta11 = eta11.to(device, dtype = torch.float)
  
    B = w1.size(0)
    with torch.no_grad():
      Theta = G(w1, lam1, eta11, batchnorm_on).cpu().detach().numpy() 
      
    return Theta
  
  if Option == "Sample":
    B1 = int(B1)
    B2 = int(B2)
    B10 = int(B10)
    
    theta_hat = 0.0
    Theta1 = 0.0
    Theta2 = 0.0
    Theta_cv_one = 0.0
    CV_err = 0.0
    Theta_stab = 0.0
    
    G.eval()  
    a1_many_sample = torch.distributions.exponential.Exponential(torch.ones(B10,S))
    one2 = torch.ones(B2,B10,S).to(device)
    one3 = torch.ones(B10,S).to(device)
  
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
        alpha1 = torch.ones(B10, S).to(device)
        theta_hat = G(alpha1, lam, eta, batchnorm_on).cpu().detach().numpy()
        
      if type1 == "DoubleBoot":
        lam = torch.ones(B10,1).to(device)
        alpha1 = a1_many_sample.sample().to(device)
        m = torch.mean(alpha1,1)
        m = m.reshape(B10,1).to(device)
        alpha1 = alpha1/m
        Theta1 = G(alpha1, lam, eta, batchnorm_on)
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
        
    return  Theta1, Theta2, Theta_cv_one, Theta_stab, CV_err, theta_hat
  
  if Option == "Train":
    n1 = float(n)
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
      optimizer = torch.optim.NAdam(G.parameters(), lr=lr0)
    optimizer.step()
    LOSS = 10000.0*torch.zeros(num_it).to(device)
    alpha = torch.ones(K0,S).to(device)
    A = torch.zeros(n, S)
    for i in range(S):
      ind = range(i*n_b,(i+1)*n_b)
      A[ind,i] = 1
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
    if model == "VC" or model == "nonpara":
      eta_cand = eta_cand.to(device)
      C1 = torch.zeros(n-1,n).to(device)
      C2 = torch.zeros(n-2,n).to(device)
      C3 = torch.zeros(n-3,n).to(device)
      C4 = torch.zeros(n-4,n).to(device)
      
      for i in range(n-2):
        C1[i,i] = 1.0
        C1[i,i+1] = -1.0
        C2[i,i] = 1.0
        C2[i,i+1] = -2.0
        C2[i,i+2] = 1.0
        
      C1[n-2,n-2] = 1.0
      C1[n-2,n-1] = -1.0
      for i in range(n-3):
        C3[i,i] = 1.0
        C3[i,i+1] = -3.0
        C3[i,i+2] = 3.0
        C3[i,i+3] = -1.0
      for i in range(n-4):
        C4[i,i] = 1.0
        C4[i,i+1] = -4.0
        C4[i,i+2] = 6.0
        C4[i,i+3] = -4.0
        C4[i,i+3] = 1.0
        
    ones000 = torch.ones(K0-U1, sub_size).to(device)  
    print(eta_on)
    sys.stdout.flush()
    alpha1 = torch.ones(1,S).to(device)
    while J == 1:
        if eta_on == 1:
          if it % 20 == 5:
            eta += 0.01*torch.randn(K0,1).to(device)
          ind_eta = np.random.choice(range(n_eta),int(K0/2)) 
          eta[range(int(K0/2)),:] = eta_cand[ind_eta].reshape(int(K0/2),1).to(device)
          eta[range(int(K0/2)),:] += 0.05*torch.randn(int(K0/2),1).to(device)

        ##############################
        lr = lr0/((it0+1.0)**lr_power)
        if sgd == 'Adam' or 'SGD':
          for param_group in optimizer.param_groups:
              param_group["lr"] = lr
        if sgd == 'RMSprop':
          scheduler.step()
          #if sgd == 'RMSprop':
          for param_group in optimizer.param_groups:
          #    param_group["lr"] *=  0.9995
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
          #weight1 = w1_sample.sample().to(device)
          weight2 = torch.distributions.gamma.Gamma(weight1, 1.0).sample().to(device)
          alpha[U1:K0, :] = weight2
          m = torch.mean(alpha,1).to(device)
          #alpha = alpha/(m.reshape(alpha.size()[0], 1))
          ONE = torch.ones(K0, S).to(device)
          #alpha = alpha.mean(1)
          #ONE[range(int(K0/10)),:] = 1.0
        if double_boot == 0 and stab_sel_on == 0 and L1pen_on == 0:
          alpha = w_full_sample.sample().to(device)
          #if it % 20 == 10:
          #  alpha[range(int(K0/2)),:] = torch.exp( torch.log(alpha[range(int(K0/2)),:]+0.00000001) + 0.01*torch.randn(int(K0/2),S).to(device))
          #alpha[range(int(K0/2),K0),:] = w_half_sample.sample().to(device)
          #if it % 50 == 0:
          #  print("SingleBoot")
        if model == "VC" or model == "nonpara":
          ONE = torch.ones(K0, S).to(device)
    
        alpha *= ONE 
        ind_one = sample(range(K0), int(K0/10))
        alpha[ind_one,:] = 1.0
        w1 = torch.matmul(alpha, A).t()
        w0 = w1[ind_sub==1.0,:]
    
        if L1pen_on == 1 or stab_sel_on == 1:
          #if it % 20 == 0:
          #  Lam = torch.exp( torch.log(Lam) + 0.01*torch.randn(K0,1).to(device))
          ind_lam = np.random.choice(range(M),int(K0)) 
          Lam[range(int(K0)),:] = torch.exp( torch.log(lam_schd[ind_lam].reshape(int(K0),1)) + 0.2*torch.randn(int(K0),1).to(device))
        #with torch.autograd.set_detect_anomaly(True):
        if model == "nonpara" or model == "VC":
          loss = 0.0
          for k1 in range(5):
            #ind_lam = np.random.choice(range(M),int(K0)) 
            #Lam[range(int(K0)),:] = torch.exp( torch.log(lam_schd[ind_lam].reshape(int(K0),1)) + 0.2*torch.randn(int(K0),1).to(device))
            ind_one = sample(range(K0), 1)
            #alpha1 = torch.ones(n,1)*w_one_sample.sample()
            #alpha1 = alpha1.to(device)
            ind_b = sample(range(S), V)
            u = w_one_sample.sample().to(device)
           #l1 = torch.distributions.exponential.Exponential(torch.ones(1)).sample().to(device)
            alpha1[:, ind_b] = u#l1[0] 
            alpha2 = torch.ones(n,1).to(device)*alpha1
            #Lam1 = Lam[ind_one,:].item()*torch.ones(n,1).to(device)
            #lam = Lam1[0,0]
            l = torch.distributions.exponential.Exponential(torch.ones(1)).sample().to(device)
            #l = 0.01*torch.exp(2.5*l)
            l = 10.0*l**3
            Lam1 = (l.item()*torch.ones(n,1)).to(device)
            w1 = torch.matmul(alpha1, A).t()
            w0 = w1[:,0].reshape(n,1)
            Theta = G(alpha2, Lam1, eta_cand, batchnorm_on)
            loss1 = Loss_func(y0, X0, Theta, eta_cand).to(device)
            loss_fit = (loss1*w0).mean()
            #d1 = C1 @ Theta
            x1 = eta_cand[range(1,n),:] - eta_cand[range(n-1),:]
            #x1 = x1.reshape(n-1,1) + 0.00001
            #D1 = d1 / x1
            #D1 *= torch.mean(x1)
            
            #d2 = D1[range(1,n-1),:] - D1[range(n-2),:]
            #x2 = eta_cand[range(2,n),:] - eta_cand[range(1,n-1),:]
            #x2 = x2.reshape(n-2,1)
            
            #x2 = x2.reshape(n-2,1) + 0.00001
            #D2 = d2 / (x2 + 0.00001)
            D2 = C2 @ Theta
            #d2 = C2 @ eta_cand
            #D2 /= torch.mean(x1).item()
            #print(D2.size())
            pen = l.item()*Penalty(D2, D).to(device) 
            #print("01983sdasfojaosdk")
            #sys.stdout.flush()
            loss_pen = pen.mean()
            #loss_pen = lam*torch.mean( torch.abs(D2) )*float(p)
            loss +=  (loss_fit + loss_pen) / 5
            loss0_pen += loss_pen.item() / 5
            
        else:
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
        
    LOSS = LOSS.cpu().detach().numpy()
    if batchnorm_on == 1:
      bn = "batchnorm on" 
    else:
      bn = "batchnorm off" 
    #eta_cand = eta_cand.cpu().detach().numpy() 
    return G, L1pen_on, double_boot, lam_schd, stab_sel_on, LOSS, K, n, p, S, hidden_size, L, K0, S, NN_type,  bn, gpu_ind#, ONE, Theta, Lam
    
    
