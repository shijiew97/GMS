#' @title
#' Loading function of Generative Multiple-purpose Sampler
#' @description
#' Loading a pre-trained generator for further training or generation
#'
#' @param save_filePy the saved .pt/ckpt file directory from GB-NPMLE function.
#' @param save_fileR the saved .RData file directory from GB-NPMLE function.
#' @param option option = "Train": load a pre-trained generator and further train;
#' option = "Sample": load a pre-trained generator and generate bootstrap samples;
#' option = "Generator": the saved generator.
#' @param gpu_ind gpu index.
#' @param num_it number of iterations for training.
#' @param B1 the number of bootstrap samples in the first level.
#' @param B2 the number of bootstrap samples in the second level.
#' @param B10 the number of the first level bootstrap samples to be
#' evaluated at each iteration. This is to reduce the amount of RAM and
#' GPU memory in storing the bootstrap samples.
#' @param eta auxiliary parameter: a scalar.
#' @param lam tuning parameter: a scalar.
#'
#' @export
#' @seealso \code{\link{GMS_Sampling}}, \code{\link{GMS}}, \code{\link{post_process}}
#' @author Minsuk Shin, Jun Liu and Shijie Wang
#' @examples
#' #Pre-training stage
#' library(reticulate)
#' set.seed(82941)
#' n = 500;p = 30;S = 100
#' bt0 = seq(-3, 3,length.out = p)
#' B1 = 5000;B2 = 1000;B10 = 100;alpha0 = 0.95;n_b2 = n/S
#' X = matrix(rnorm(n*p),n,p)
#' for(j in 1:p){X[,j] = (X[,j] + rnorm(n))/2}
#' mu = X%*%bt0
#' prob = 1/(1+exp(-1*mu))
#' y = matrix(rbinom(n,1,prob), n, 1)
#' fit_GMS = GMS(X, y, model="logistic", type="DoubleBoot", num_it=250,
#' lr_power=0.2, L=4, S=100, lr0=0.0001, sgd="Adam", hidden_size=1000,
#' NN_type="WM-MLP", save_on=1, save_path="")
#'
#'
#' #Option: Train the saved model

#' fit_load = GMS_Loading(save_filePy, save_fileR, option="Train", gpu_ind=0,
#' num_it=100, B1=3000, B2=100, eta=NULL)
#' #Option: Generator for theta_hat
#' theta_hat1 = GMS_Loading(save_filePy, save_fileR, option="Generator",
#' lam=1, gpu_ind=0,
#' num_it=100, B1=3000, B2=100, eta=NULL)
#'
#' #Option: Generation for bootstrap samples
#' sample_load = GMS_Loading(save_filePy, save_fileR, option="Sample", gpu_ind=0,
#' num_it=100, B1=3000, B2=100, B10=100, eta=NULL)
#' res = post_process(sample_load, alpha=0.95, theta_hat=theta_hat1)
GMS_Loading <- function(save_filePy=NULL, save_fileR=NULL, option=NULL,
                        gpu_ind, num_it, B1=3000, B2=100, B10=NULL, eta=NULL, lam=1
                        ){
  require(reticulate)

  if(is.null(option) == TRUE){stop("Option is missing!")}
  if(is.null(save_filePy) == TRUE){stop("Generator file (.py) path is missing!")}
  if(is.null(save_fileR) == TRUE){stop("Generator file (.RData) is missing!")}
  if(is.null(eta) == TRUE){eta = 0.5}
  if(option == "Train"){B1=B2=B10=100}else{num_it=100}

  out_GMS = readRDS(save_fileR)
  model = as.character(out_GMS[[3]])
  custom_loss_file = unlist(out_GMS[[4]], use.names=F)[1]
  custom_penalty_file = unlist(out_GMS[[4]], use.names=F)[2]
  NN_setting = out_GMS[[5]]
  NN_type = as.character(NN_setting[1])
  S = as.numeric(NN_setting[2])
  L = as.numeric(NN_setting[3])
  hidden_size = as.numeric(NN_setting[4])
  lr0 = as.numeric(NN_setting[5])
  lr_power = as.numeric(NN_setting[6])
  batchnorm_on = as.numeric(NN_setting[7])
  p = as.numeric(NN_setting[8])
  cv_k = as.numeric(NN_setting[9])
  sub_size = as.numeric(NN_setting[10])
  K0 = as.numeric(NN_setting[11])
  sgd = as.character(NN_setting[12])
  verb = as.numeric(NN_setting[13])
  penalty = as.character(NN_setting[14])
  n = as.numeric(NN_setting[15])
  eta_on = as.numeric(NN_setting[16])
  n_eta = as.numeric(NN_setting[17])
  #C = NN_setting[18]
  cv_on = as.numeric(NN_setting[19])
  type = as.character(NN_setting[20])
  X = matrix(unlist(NN_setting[21]),n,p)
  y = matrix(unlist(NN_setting[22]),n,1)
  lam_schd = as.vector(out_GMS[[7]])
  eta_cand = as.vector(out_GMS[[9]])
  intercept = out_GMS[[10]]
  penalty_type = penalty
  ind_perm = as.vector(out_GMS[[12]])


  if( is.null(custom_penalty_file) == TRUE ){
    if(NN_type!= "Linear" & NN_type!= "WML-MLP" &
       NN_type!= "MLP+Linear" & NN_type!= "Linear"
    ){
      pr = paste("The specification of the type of NN, '",NN_type,"', is wrong!",sep="")
      print(pr)
      print("The available NN types are 'MLP', 'WML-MLP', 'MLP+Linear', and 'Linear'.")
    }
  }

  D = 0.0
  if(is.null(X) == TRUE){D = 0.0}
  if(is.vector(X) == TRUE){X = matrix(X,n,p)}
  if(is.vector(y) == TRUE){y = matrix(y,n,1)}

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

  w = matrix(1,1,S)
  lam = lam*matrix(1,dim(w)[1],1)
  eta000 = eta*matrix(1,dim(w)[1],1)
  lam0 = r_to_py(lam, convert = FALSE)
  w0 = r_to_py(w, convert = FALSE)
  eta00 = r_to_py(eta000, convert = FALSE)

  M = length(lam_schd)
  lam_schd = matrix(lam_schd, M, 1)
  Option0 = r_to_py(option)
  Save_file0 = r_to_py(save_filePy, convert = FALSE)
  y0 = r_to_py(y, convert = FALSE)
  X0 = r_to_py(X, convert = FALSE)
  S0 = r_to_py(S, convert = FALSE)
  hidden_size0 = r_to_py(hidden_size, convert = FALSE)
  L0 = r_to_py(L)
  NN_type0 = r_to_py(NN_type, convert = FALSE)
  sub_size0 = r_to_py(sub_size, convert = FALSE)
  cv_K0 = r_to_py(cv_k, convert = FALSE)
  n0 = r_to_py(n, convert = FALSE)
  p0 = r_to_py(p, convert = FALSE);
  double_boot0 = r_to_py(double_boot, convert = FALSE)
  cv_on0 = r_to_py(cv_on, convert = FALSE)
  lam_schd0 = r_to_py(lam_schd, convert = FALSE)
  lr_power0 = r_to_py(lr_power)
  lr0 = r_to_py(lr0, convert = FALSE)
  gpu_ind0 = r_to_py(gpu_ind, convert = FALSE)
  verb0 = r_to_py(verb, convert = FALSE)
  num_it0 = r_to_py(num_it, convert = FALSE)
  stab_sel_on0 = r_to_py(stab_sel_on, convert = FALSE)
  K00 = r_to_py(K0, convert = FALSE)
  batchnorm_on0 = r_to_py(batchnorm_on, convert = FALSE)
  sgd0 = r_to_py(sgd, convert = FALSE)
  eta_cand0 = r_to_py(eta_cand, convert = FALSE)
  D0 = r_to_py(D, convert = FALSE)
  type0 = r_to_py(type, convert = FALSE)
  penalty_type0 = r_to_py(penalty, convert = FALSE)
  #C0 = r_to_py(C, convert = FALSE)
  model0 = r_to_py(model, convert = FALSE)
  eta0 = r_to_py(eta, convert = FALSE)
  sub_size0 = r_to_py(sub_size, convert = FALSE)
  eta_on0 = r_to_py(eta_on, convert = FALSE)

  if(is.null(B10)){B10 = B1}
  y_py = r_to_py(y, convert = FALSE)
  X_py = r_to_py(X, convert = FALSE)
  B1_py = r_to_py(B1, convert=FALSE)
  B2_py = r_to_py(B2, convert=FALSE)
  B10_py = r_to_py(B10, convert=FALSE);
  lam_schd_py = r_to_py(lam_schd, convert = FALSE)

  if( is.null(custom_loss_file) == TRUE){
    if(model == "linear"){code_loss <- paste(system.file(package="GMS"), "Loss_Linear.py", sep="/")}
    if(model == "logistic"){code_loss <- paste(system.file(package="GMS"), "Loss_Logit.py", sep="/")}
    if(model == "LAD"){code_loss <- paste(system.file(package="GMS"), "Loss_LAD.py", sep="/")}
    if(model == "quantile"){code_loss <- paste(system.file(package="GMS"), "Loss_Quantile.py", sep="/")}
    if(model == "VC"){code_loss <- paste(system.file(package="GMS"), "Loss_VC.py", sep="/")}
    if(model == "nonpara"){code_loss <- paste(system.file(package="GMS"), "Loss_nonpara.py", sep="/")}
    reticulate::source_python(code_loss)#, envir = NULL,convert = FALSE)

  }else{
    model = NULL
    reticulate::source_python(custom_loss_file)#, envir = NULL,convert = FALSE)
  }

  if( is.null(custom_penalty_file) == TRUE ){
    if(penalty == "L1") code_pen <- paste(system.file(package="GMS"), "pen_L1.py", sep="/")
    if(penalty == "L2") code_pen <- paste(system.file(package="GMS"), "pen_L2.py", sep="/")
  }else{
    code_pen <- paste(system.file(package="GMS"), custom_penalty_file, sep="/")
  }

  reticulate::source_python(code_pen)

  Loss_func = py$Loss_func
  Penalty = py$Penalty

  code_Loading <- paste(system.file(package="GMS"), "GMS_load_function.py", sep="/")
  #code_Loading = "E:/Dropbox/Shijie/GMS/inst/GMS_load_function.py"
  reticulate::source_python(code_Loading)#, envir = NULL,convert = FALSE)


  if(option == "Train"){

    pmt = proc.time()[3]
    fit_load = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                        sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                        cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                        B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                        Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
    time_tr = proc.time()[3]-pmt

    print("Training Done!")
    print("######################################################")
    print(paste("Time for training of G: ",round(time_tr,2), " seconds"))
    print("######################################################")
    type_save = list(type = type)
    if( type == "CV" | type == "NCV"){
      type_save = list(type = type, cv_K = cv_K)
    }
    LOSS = fit_load[[6]]
    custom = list(custom_loss_file = custom_loss_file, custom_penalty_file = custom_penalty_file)
    NN_setting = list(NN_type = NN_type, S = S, L = L, hidden_size = hidden_size,
                      lr0 = lr0, lr_power = lr_power, batchnorm_on = batchnorm_on,
                      p = p, cv_K = cv_k, sub_size = sub_size, K0 = K0, sgd = sgd,
                      verb = verb, penalty = penalty, n = n, eta_on = eta_on,
                      n_eta = n_eta, C = C, cv_on = cv_on)
    out_load = list(fit_load = fit_load, type = type_save, model = model, custom = custom, NN_setting = NN_setting, time_tr = time_tr,
                   lam_schd = as.vector(lam_schd), LOSS = LOSS,
                   eta_cand=eta_cand, intercept=intercept, penalty_type=penalty_type, ind_perm = ind_perm,
                   X = X, Y = y)
  }

  if(option == "Sample"){
    A1 = B1/B10

    Theta1 = Theta2 = Theta_cv = Theta_stab = CV_err = theta_hat = NULL

    pmt = proc.time()[3]
    if(type == "CV"){
      samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                         sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                         cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                         B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                         Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
      #samples = py_to_r(samples)
      Theta_cv = samples[[3]]
      CV_err = samples[[5]]
    }
    if(type == "NCV"){
      samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                         sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                         cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                         B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                         Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
      #samples = py_to_r(samples)
      Theta_cv = samples[[3]]
      CV_err = samples[[5]]
    }
    if(type == "Boot-CV"){
      K = cv_K
      CV_err_boot = array(0,dim=c(M,K,B1))
      for(m in 1:A1){
        ind = ((m-1)*B10+1):(m*B10)
        samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                           sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                           cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                           B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                           Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
        #Theta_cv0 = samples[[3]]
        CV_err0 = samples[[5]]
        CV_err_boot[,,ind] = CV_err0
        pr = paste("",round(100*m/A1,1),"% completed")
        cat(pr, "\n")
      }
      type = "CV"
      samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                         sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                         cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                         B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                         Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
      #samples = py_to_r(samples)
      Theta_cv0 = samples[[3]]
      CV_err0 = samples[[5]]
      type = "Boot-CV"
    }

    if(type == "DoubleBoot"){
      Theta1 = matrix(0,B1,p)
      Theta2 = array(0,dim=c(B2,B1,p))
      for(m in 1:A1){
        ind = ((m-1)*B10+1):(m*B10)
        samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                           sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                           cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                           B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                           Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
        #samples = py_to_r(samples)
        Theta1[ind,] = samples[[1]]
        Theta2[,ind,] = samples[[2]]
        cat(paste(round(100*m/A1,1),"%",sep=""), sep=" ")
        cat(" | ")
      }
      theta_hat = samples[[6]][1,]
    }
    if(type == "StabSel"){
      Theta_stab = array(0,dim=c(M,B1,p))
      for(m in 1:A1){
        ind = ((m-1)*B10+1):(m*B10)
        samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                           sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                           cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                           B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                           Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
        #samples = py_to_r(samples)
        Theta_stab[,ind,] = samples[[4]]
        cat(paste(round(100*m/A1,1),"%",sep=""))
        cat(" | ")
      }
    }
    if(type == "SingleBoot" | type == "Tuning"){
      Theta1 = matrix(0,B1,p)
      for(m in 1:A1){
        ind = ((m-1)*B10+1):(m*B10)
        samples = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                           sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                           cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                           B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                           Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
        #samples = py_to_r(samples)
        Theta1[ind,] = samples[[1]]
        cat(paste(round(100*m/A1,1),"%",sep=""))
        cat(" | ")
      }
    }
    time_samp = proc.time()[3]- pmt
    #  reticulate::source_python(code_Sampling, envir = NULL,convert = FALSE)
    theta_hat = samples[[6]]
    if(type == "SingleBoot" |  type == "Tuning"){
      total = B1
      lam_schd = NULL
    }
    if(type == "DoubleBoot"){
      total = B1*B2
      lam_schd = NULL
    }
    if(type == "Boot-CV"){
      total = cv_k*M*B1
    }

    if(type == "CV"){
      total = cv_k*M
    }
    if(type == "NCV"){
      a = cv_k
      total = a*(a-1)*M
    }
    if(type == "StabSel"){
      total = B1*M
    }

    #  total = dim(Theta1)[1]*dim(Theta2)[1]
    print("Generation Done!")
    print("######################################################")
    #print(paste("The first level: ", dim(Theta1_GMS)[1]," bootstrap samples are generated.",sep=""))
    #print(paste("The second level: ", dim(Theta2_GMS)[1]," bootstrap samples are generated.",sep=""))
    #print(paste("The third level: ", dim(Theta3_GMS)[1]," bootstrap samples are generated.",sep=""))
    print("------------------------------------------------------")
    print(paste("Type: ", type, sep=""))
    print(paste("Total: ", total," estimators are evaluated.",sep=""))
    print("######################################################")
    #print(paste("Time for training: ",round(time_tr,2), " seconds"))
    print(paste("Time for generation: ",round(time_samp,2), " seconds"))
    print("######################################################")
    out_load = list( Theta1 = Theta1, Theta2 = Theta2, Theta_cv = Theta_cv, CV_err = CV_err, Theta_stab = Theta_stab, lam_schd = lam_schd,
                time_samp = time_samp, theta_hat = theta_hat, type=type, eta = eta, n = n)
    if( type == "Boot-CV" ){
      out_load = list(Theta1 = Theta1, Theta2 = Theta2, Theta_cv = Theta_cv, CV_err_boot = CV_err_boot, CV_err0 = CV_err0, Theta_cv0 = Theta_cv0,
                 lam_schd = lam_schd,time_samp = time_samp,  type=type, eta = eta, n = n)
    }
  }

  if(option == "Generator"){
    out_load = GMS_load(Option0, Save_file0, gpu_ind0, S0, n0, p0, batchnorm_on0,
                        sub_size0, K00, NN_type0, verb0, L0, hidden_size0, cv_K0, stab_sel_on0,
                        cv_on0, double_boot0, lr_power0, lr0, num_it0, lam_schd_py, y_py, X_py,
                        B1_py, B2_py, B10_py, type0, eta0, eta_on0, model0, eta_cand0,
                        Loss_func, Penalty, penalty_type0, sgd0, w0, lam0, eta00)
  }
  return(out_load)
}
