def Loss_func(y, X, Theta, eta):
    #c = X*Theta
    #n = X.size()[0]
    #b = c.sum(1).reshape(n,1)
    out =  (y - Theta)**2
    return out




