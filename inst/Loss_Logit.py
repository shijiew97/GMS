def Loss_func(y, X, Theta, eta):
  c = torch.matmul(X,Theta.t())
  c = torch.clamp(c, -50.0, 50.0) 
  out =  (1-y)*c + torch.log(1.0+torch.exp(-c))
  return out
