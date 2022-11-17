def Penalty(Theta, D):
  k0 = Theta.size()[0]
  pen = (Theta**2).sum(1).reshape(k0,1)
  return pen



