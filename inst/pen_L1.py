def Penalty(Theta, D):
  k0 = Theta.size()[0]
  pen = torch.abs(Theta).sum(1).reshape(k0,1)
  return pen



