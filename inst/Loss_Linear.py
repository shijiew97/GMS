def Loss_func(y, X, Theta, eta):
  c = torch.matmul(X,Theta.t())
  out =  (y - c)**2
  return out
