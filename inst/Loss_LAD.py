def Loss_func(y, X, Theta, eta):
  c = torch.matmul(X,Theta.t())
  out =  torch.abs(y - c)
  return out
