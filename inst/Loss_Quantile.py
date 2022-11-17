def Loss_func(y, X, Theta, eta):
    eta = torch.clip(eta, 0.001, 0.999)
    c = torch.matmul(X, Theta.t())
    b = y - c
    out =  (1.0- eta.t())*nn.ReLU()(-b) + eta.t()*nn.ReLU()(b)
    #print("sdaf")
    #sys.stdout.flush()
    return out




