#' @importFrom reticulate py_module_available
#' @importFrom reticulate source_python
#' @importFrom reticulate py_available
#' @title
#' Generative Multiple-purpose Sampler
#'
#' @description
#' Train the generator of GMS. You need to install python (>=3.7) and pytorch in advance.
#' For the details of installation, visit https://pytorch.org/get-started/locally/.
#' This R package is a wrapepr of python (using pytorch) programs via "reticulate" R package.
#'
#' @param X predictor variable.
#' @param y response variable.
#' @param eta_cand a candidate set of eta.
#' @param model the model of interest. Linear model: "linear"; logistic regression models: "logistic"; least absolute deviation regression: "LAD" quantile regression: "quantile".
#' @param type 	type of GMS. "DoubleBoot": double bootstrap; "CV": cross-validation; "StabSel": stability selection. "SingleBoot" can be used, but "DoubleBoot" type also learns the single bootstrap. We recommend to use "DoubleBoot", instead of "SingleBoot".
#' @param cv_K the number of folds for the CV.
#' @param sub_size subsampling size when the data size is extremely large.
#' @param S the number of subgroups in the bootstrap.
#' @param K0 update sample size at each iteration.
#' @param NN_type the type of neural network generator. "MLP": feed-forwarding NN; "Hadamard": feed-forwarding NNHadamard producted with the bootstrap weights at the last layer; "lowrank": low rank parameterization.
#' @param lr0 learning rate.
#' @param lr_power power rate of learning decay.
#' @param num_it the number of iterations.
#' @param hidden_size the number of hidden neurons at each layer.
#' @param L the number of hidden layers.
#' @param gpu_ind gpu index to run the computation. Useful under multiple GPU environment.
#' @param custom_loss_file python file that defines a customized loss function of interest. The name of the loss function should be defined as "Loss_func".
#' It should input X, y, generated theta's, and the dimension of the generated theta should be K0 X p, where K0 is the number of Monte Carlo samples at each iteration.
#' The ouput of the loss function should be n X K0. Each column is individual losses of observations
#' for a bootstrap weight; w_i*L(theta; y_i, X_i)_i=1,...,n. An example of a linear regression is \cr
#' def Loss_func(y, X, Theta): c = torch.matmul(X,Theta.t()) out = (y - c)**2 return out \cr
#' For a logistic regression, \cr
#' def Loss_func(y, X, Theta): c = torch.matmul(X,Theta.t()) c = torch.clamp(c, -50.0, 50.0) out = (1-y)*c + torch.log(1.0+torch.exp(-c)) return out.
#' @param custom_penalty_file python file that defines a customized penalty
#' function of interest. Default is L1 penalty. The name of the penalty
#' function should be defined as "Penalty", and the penalty funciton
#' inputs a K0 X p dimensional Theta and outputs K0 X 1 dimensional
#' penalty values. For example of L1 penalty, \cr
#' def Penalty(Theta): k0 = Theta.size()[0] pen = torch.abs(Theta).sum(1).reshape(k0,1) return pen
#' @param lam_schd a candidate set of the tuning parameter of the penalty.
#' @param save_on save the trained generator function. 1: save; 0: no save. Default is 0.
#' @param save_path the directory path where the trained generator function will be saved.
#' @param batchnorm_on batch normalization is used.
#' @param verb verb=1: print all information during training every 100 iterations; verb=0: minimal information.
#'
#' @usage
#' GMS(X = NULL, y, eta_cand = NULL, model, p = NULL, type,
#' cv_K = 10, sub_size = NULL, S = 100, K0 = 200,  NN_type="WML-MLP",
#' lr0 = 0.0005, lr_power = 0.2, num_it = 10000, hidden_size = NULL,
#' L = 4, gpu_ind = 0, custom_loss_file = NULL, custom_penalty_file = NULL,
#' lam_schd = NULL, save_on = 0, save_path = NULL, batchnorm_on = 0,
#' sgd = "Adam", verb = 2, intercept = FALSE, penalty = "L1")
#'
#' @author Minsuk Shin, Jun Liu and Shijie Wang
#' @seealso \code{\link{GMS_Sampling}}, \code{\link{post_process}}, \code{\link{GMS_Loading}}, \code{\link{generator}}
#' @export
#' @examples
#' #### linear regression example
#' library(reticulate)
#' set.seed(82941)
#' n = 500;p = 50
#' bt0 = seq(-1,1,length.out = p)
#' X = matrix(rnorm(n*p),n,p)
#' mu0 = crossprod(t(X),bt0)
#' y = mu0 + rnorm(n)
#' fit = lm(y~0+X)
#' theta_hat = fit$coefficients
#' ##############################
#' #### Training steps
#' #fit_GMS = GMS(X, y, model="linear", type="DoubleBoot", NN_type="WM-MLP")
#' #samples_GMS = GMS_Sampling(fit_GMS, B1 = 1000, B10 = 500, X = X, y = y)
#' #res = post_process(samples_GMS, theta_hat = theta_hat, thre=0.001)
#' #par(mfrow=c(2,2),mai=c(0.4,0.4,0.1,0.1))
#' #for(k in 1:4){
#' #  plot(1:p, type="n", ylim=c(-2.5,2))
#' #  CI = res[[k+1]]
#' #  points(bt0, pch=4,col="blue")
#' #  for(j in 1:p){
#' #    lines(rep(j,2), c(CI[j,1],CI[j,2]), col="red",lwd=2)
#' #  }
#' #  cov = 100*length(which(CI[,1]<bt0 & CI[,2]>bt0))/p
#' #  text(p/2,-2, paste(cov,"%",sep=""))
#' #}
#' #### Logistic regression example
#' library(reticulate)
#' set.seed(82941)
#' n = 500;p = 30;S = 100
#' bt0 = seq(-3, 3,length.out = p)
#' B1 = 3000;B2 = 100;B10 = 100;alpha0 = 0.95;n_b2 = n/S
#' X = matrix(rnorm(n*p),n,p)
#' for(j in 1:p){X[,j] = (X[,j] + rnorm(n))/2}
#' mu = X%*%bt0
#' prob = 1/(1+exp(-1*mu))
#' y = matrix(rbinom(n,1,prob), n, 1)
#' fit_GMS = GMS(X, y, model="logistic", type="DoubleBoot", num_it=25000,
#' lr_power=0.2, L=4, S=100, lr0=0.0001, sgd="Adam", hidden_size=1000,
#' NN_type="WM-MLP")
#' samples_GMS = GMS_Sampling(fit_GMS, B1=B1, B2=B2, B10=B10, X=X, y=y,
#' type="DoubleBoot", gpu_ind=0)
#' theta_hat = generator(fit_GMS, w=matrix(1,1,S), verb=0)
#' res = post_process(samples_GMS, alpha=alpha0, theta_hat=theta_hat)
GMS <- function(X = NULL, y, eta_cand = NULL, model, p = NULL, type = "DoubleBoot", cv_K = 10, sub_size = NULL, S = 100, K0 = 200,  NN_type="WM-MLP",
                lr0 = 0.0005, lr_power = 0.2, num_it = 20000, hidden_size = NULL,
                L = 3, gpu_ind = 0, custom_loss_file = NULL, custom_penalty_file = NULL, lam_schd = NULL,
                save_on = 0, save_path = NULL, batchnorm_on = 0, sgd = "Adam", verb = 2, intercept = FALSE, penalty = "L1"){
  # NN-type: "MLP"; "MLP+Linear"; "Linear"; "WML-MLP"
  #require(reticulate)
  # type: "CV"; "NCV"; "DoubleBoot"; "StabSel"; "SingleBoot"
  require(reticulate)
  if( is.null(custom_penalty_file) == TRUE ){
    if(NN_type!= "Linear" & NN_type!= "WM-MLP" &
       NN_type!= "MLP+Linear" & NN_type!= "Linear"
    ){
      pr = paste("The specification of the type of NN, '",NN_type,"', is wrong!",sep="")
      print(pr)
      print("The available NN types are 'MLP', 'WM-MLP', 'MLP+Linear', and 'Linear'.")
    }
  }
  D = 0.0
  if(is.null(X) == T ){
    D = 0.0
  }
  n = length(y)
  ind_perm = sample(1:n,n)
  #y = y[ind_perm]
  y = matrix(y,n,1)
  if(is.null(X) == T ){
    X = matrix(0,10,1)
  }else{
    #X = X[ind_perm,]
    if(intercept == TRUE){
      n = nrow(X)
      X = cbind(rep(1,n),X)
      p = ncol(X)
    }
    if(is.vector(X) == T){
      X = matrix(X,n,p)
    }
  }
  if( model == "VC" | model == "nonpara"){
    #p = 1
    #type="CV"
    eta_on = 0
    if(is.null(eta_cand) == TRUE){
      print("ERROR: Temporal variable is missing!")
    }
    #eta_cand = eta_cand[ind_perm]
    eta_cand = matrix(eta_cand,n,1)
  }else{
    if(is.null(eta_cand) == T ){
      eta_cand = matrix(0.5,10,1)
      n_eta = length(eta_cand)
      eta_on = 0
    }else{
      eta_on = 1
      n_eta = length(eta_cand)
    }
  }

  n = nrow(y)
  #
  if(is.null(p) == T){
    p = ncol(X)
    print("'p' is set to the dimension of X")
  }
  if( is.null(custom_loss_file) == TRUE){
    print(paste("Model:",model))
    print(paste("Type:",type))
  }else{
    print(paste("Model: Customized model,",custom_loss_file))
    print(paste("Type:",type))
  }
  today = Sys.Date()
  if(is.null(save_path)==TRUE){
    path = ""
  }
  save_file = paste(save_path,"G_",today,".ckpt",sep="")
  save_Rfile = paste(save_path,"out_",today,".RData",sep="")
  save_struc = paste(save_path,"structure_",today,".txt",sep="")
  #cond_B1 = length(intersect(ls(envir=.GlobalEnv),"B1"))>0
  #cond_B2 = length(intersect(ls(envir=.GlobalEnv),"B2"))>0
  #cond_B3 = length(intersect(ls(envir=.GlobalEnv),"B3"))>0
  if(is.null(hidden_size)==TRUE){
    hidden_size = max(1000,p*3)
  }
  if(is.null(lam_schd)==TRUE){
    lam_schd = exp(seq(-5, -1,length.out = 100))
  }
  if(missing(NN_type)){
    NN_type = "WM-MLP"
  }
  #if(missing(low)){
  #  low = 5
  #}
#  if(NN_type!= "MLP" & NN_type!= "WM-MLP" &
#     NN_type!= "MLP+Linear" & NN_type!= "Linear"
#     ){
#    pr = paste("The specification of the type of NN, '",NN_type,"', is wrong!",sep="")
#    print(pr)
#    print("The available NN types are 'MLP', 'WML-MLP', 'MLP+Linear', and 'Linear'.")
#  }
  if(is.null(sub_size)==TRUE){
    sub_size = S
  }
  cv_on = 0
  double_boot = 0
  stab_sel_on = 0
  if(type == "CV"){
    cv_on = 1
    double_boot = 0
    stab_sel_on = 0
  }
  if(type == "DoubleBoot"){
    cv_on = 0
    double_boot = 1
    stab_sel_on = 0
  }
  if(type == "SingleBoot"){
    cv_on = 0
    double_boot = 0
    stab_sel_on = 0
  }
  if(type == "StabSel"){
    cv_on = 0
    double_boot = 0
    stab_sel_on = 1
  }
  if( type != "CV" & type != "DoubleBoot" &
      type != "StabSel" & type != "SingleBoot" ){
    print("The sepcified type argument is wrong!")
  }
  if( sgd != "Adam" & sgd != "RMSprop"){
    print("Available SGD algorithms are 'Adam' or 'RMSprop'")
  }

  NN_setting = list(NN_type = NN_type, S = S, L = L, hidden_size = hidden_size,
                    lr0 = lr0, lr_power = lr_power, batchnorm_on = batchnorm_on,
                    p = p, cv_K = cv_K, sub_size = sub_size, K0 = K0, sgd = sgd,
                    verb = verb, penalty = penalty, n = n, eta_on = eta_on,
                    n_eta = n_eta, C = C, cv_on = cv_on, type = type, x = X,
                    Y = y)

  M = length(lam_schd)
  lam_schd = matrix(lam_schd, M, 1)
  y = r_to_py(y, convert = FALSE); X = r_to_py(X, convert = FALSE); S = r_to_py(S, convert = FALSE); hidden_size = r_to_py(hidden_size, convert = FALSE)
  L = r_to_py(L); NN_type = r_to_py(NN_type, convert = FALSE); sub_size = r_to_py(sub_size, convert = FALSE); cv_K = r_to_py(cv_K, convert = FALSE)
  n = r_to_py(n, convert = FALSE); p = r_to_py(p, convert = FALSE);
  double_boot = r_to_py(double_boot, convert = FALSE); cv_on = r_to_py(cv_on, convert = FALSE); lam_schd = r_to_py(lam_schd, convert = FALSE); lr_power = r_to_py(lr_power)
  lr0 = r_to_py(lr0, convert = FALSE); gpu_ind = r_to_py(gpu_ind, convert = FALSE); save_file = r_to_py(save_file, convert = FALSE); save_struc = r_to_py(save_struc, convert = FALSE)
  verb = r_to_py(verb, convert = FALSE); num_it = r_to_py(num_it, convert = FALSE); save_on = r_to_py(save_on, convert = FALSE)
  stab_sel_on = r_to_py(stab_sel_on, convert = FALSE); K0 = r_to_py(K0, convert = FALSE);batchnorm_on = r_to_py(batchnorm_on, convert = FALSE)
  sgd = r_to_py(sgd, convert = FALSE); eta_cand = r_to_py(eta_cand, convert = FALSE)
  D = r_to_py(D, convert = FALSE); type = r_to_py(type, convert = FALSE)
  penalty_type = r_to_py(penalty, convert = FALSE)
  C0 = r_to_py(C, convert = FALSE)

  have_torch <- reticulate::py_module_available("torch")
  have_random <- reticulate::py_module_available("random")
  have_numpy <- reticulate::py_module_available("numpy")
  if (!have_torch)
    print("Pytorch is not installed!")
  if (!have_random)
    print("random is not installed!")
  if (!have_numpy)
    print("numpy is not installed!")
  if( is.null(custom_loss_file) == TRUE){
    if(model == "linear"){code_loss <- paste(system.file(package="GMS"), "Loss_Linear.py", sep="/")}
    if(model == "logistic"){code_loss <- paste(system.file(package="GMS"), "Loss_Logit.py", sep="/")}
    if(model == "LAD"){code_loss <- paste(system.file(package="GMS"), "Loss_LAD.py", sep="/")}
    #if(model == "cox"){code_loss <- paste(system.file(package="GMS"), "Loss_cox.py", sep="/")}
    if(model == "quantile"){code_loss <- paste(system.file(package="GMS"), "Loss_Quantile.py", sep="/")}
    if(model == "VC"){code_loss <- paste(system.file(package="GMS"), "Loss_VC.py", sep="/")}
    if(model == "nonpara"){code_loss <- paste(system.file(package="GMS"), "Loss_nonpara.py", sep="/")}
    reticulate::source_python(code_loss)#, envir = NULL,convert = FALSE)

  }else{
    model = NULL
    reticulate::source_python(custom_loss_file)#, envir = NULL,convert = FALSE)
  }
  code_train <- paste(system.file(package="GMS"), "GMS_train_function.py", sep="/")
  if( is.null(custom_penalty_file) == TRUE ){
    if(penalty == "L1") code_pen <- paste(system.file(package="GMS"), "pen_L1.py", sep="/")
    if(penalty == "L2") code_pen <- paste(system.file(package="GMS"), "pen_L2.py", sep="/")
  }else{
    code_pen <- paste(system.file(package="GMS"), custom_penalty_file, sep="/")
  }
  #if( penalty_on == 0){
  #  code_pen <- paste(system.file(package="GMS"), "no_pen.py", sep="/")
  #}
  reticulate::source_python(code_pen)#, envir = NULL,convert = FALSE)
  reticulate::source_python(code_train)
#  GMS_train = py_to_r(GMS_train)
  pmt = proc.time()[3]
  Loss_func = py$Loss_func
  Penalty = py$Penalty
  #y, X, S1, hidden_size1, L1, NN_type1, sub_size1, cv_K1, K01, n1, p1,
  #double_boot1, cv_on1, stab_sel_on1, lam_schd1, lr_power1, lr01, num_it1, gpu_ind1, save_on1,
  #save_file1, save_struc1, batchnorm_on1, verb1, sgd, eta_cand, Loss_func, Penalty, eta_on, model):

  fit_GMS = GMS_train(y, X, S, hidden_size, L, NN_type, sub_size, cv_K, K0, n, p,
                      double_boot, cv_on, stab_sel_on, lam_schd, lr_power, lr0, num_it,
                      gpu_ind, save_on, save_file, save_struc, batchnorm_on, verb, sgd,
                      eta_cand, Loss_func, Penalty, eta_on, model, D, penalty_type, type)
  time_tr = proc.time()[3]-pmt
  print("Training Done!")
  print("######################################################")
  print(paste("Time for training of G: ",round(time_tr,2), " seconds"))
  print("######################################################")
  type_save = list(type = type)
  if( type == "CV" | type == "NCV"){
    type_save = list(type = type, cv_K = cv_K)
  }
  eta_cand = py_to_r(eta_cand)
  LOSS = fit_GMS[[6]]
  custom = list(custom_loss_file = custom_loss_file, custom_penalty_file = custom_penalty_file)
  out_GMS = list(fit_GMS = fit_GMS, type = type_save, model = model, custom = custom, NN_setting = NN_setting, time_tr = time_tr,
                 lam_schd = as.vector(lam_schd), LOSS = LOSS,
                 eta_cand=eta_cand, intercept=intercept, penalty_type=penalty_type, ind_perm = ind_perm)

  #G, L1pen_on, double_boot, lam_schd, stab_sel_on, LOSS, K, n, p, S, hidden_size, L, K0, S, NN_type,  bn, gpu_ind#, ONE, Theta, Lam
  if( save_on == 1 ){
    #save(out_GMS, file = save_Rfile)
    saveRDS(out_GMS, file=save_Rfile)
  }
  return( out_GMS )
}
