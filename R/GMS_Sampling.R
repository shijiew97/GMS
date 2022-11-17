#' Sampling Function of Generative Bootstrap Sampler
#'
#' @description
#' Generate bootstrap sample from the trained generator of GBS.
#'
#' @param fit_GMS an GMS object evaluated by "GMS" function.
#' @param B1 the number of bootstrap samples in the first level.
#' @param B2 the number of bootstrap samples in the second level.
#' @param B10 the number of the first level bootstrap samples to be
#' evaluated at each iteration. This is to reduce the amount of RAM and
#' GPU memory in storing the bootstrap samples.
#' @param lam_schd a candidate set of tuning parameter for CV and
#' stability selection. When the value is not specified, the candidate set
#' stored in "fit_GBS" will be used.
#' @param X the predictor used in training the generator.
#' This should be given to evaluate a CV.
#' @param y the response used in training the generator.
#' This should be given to evaluate a CV.
#' @param gpu_ind gpu index to run the computation. Useful under multiple GPU environment.
#'
#' @usage
#' GMS_Sampling(fit_GMS, B1=3000, B2=100, B10=NULL, lam_schd = NULL,
#' X=NULL, y=NULL, gpu_ind = 0, type = NULL, eta = NULL)
#'
#' @author Minsuk Shin, Jun Liu and Shijie Wang
#' @seealso \code{\link{post_process}}, \code{\link{GMS}}, \code{\link{GMS_Loading}}, \code{\link{generator}}
#' @export
#' @examples
#' #samples_GBS = GBS_Sampling(fit_GBS, X = X, y = y)
GMS_Sampling <- function(fit_GMS, B1=3000, B2=100, B10=NULL, lam_schd = NULL, X=NULL, y=NULL, gpu_ind = 0, type = NULL, eta = NULL){
  #require(reticulate)
  if(is.null(B10)){
    B10 = B1
  }
  if(is.null(eta) == T ){
    eta = 0.5
  }

  if(is.null(y) == FALSE){
    if( is.vector(y) == T ){
      y = matrix(y,n,1)
    }
    y_py = r_to_py(y, convert = FALSE)
  }else{
    y_py = r_to_py(matrix(0,1,1), convert = FALSE)
  }
  intercept = fit_GMS[[10]]
  if(is.null(X) == FALSE){
    if(intercept == TRUE){
      n = nrow(X)
      X = cbind(rep(1,n),X)
      p = ncol(X)
    }
    X_py = r_to_py(X, convert = FALSE)
  }else{
    X_py = r_to_py(matrix(0,1,1), convert = FALSE)
  }
  if(is.null(lam_schd)==FALSE){
    lam_schd = matrix(lam_schd, length(lam_schd),1)
  }else{
    lam_schd = fit_GMS[[7]]
    lam_schd = matrix(lam_schd, length(lam_schd),1)
  }
  M = length(lam_schd)
  eta_cand = fit_GMS[[9]]

  have_torch <- reticulate::py_module_available("torch")
  have_random <- reticulate::py_module_available("random")
  have_numpy <- reticulate::py_module_available("numpy")
  if (!have_torch)
    print("Pytorch is not installed!")
  if (!have_random)
    print("random is not installed!")
  if (!have_numpy)
    print("numpy is not installed!")
  #print("Training Done!")
  print("Generation Starts!")
  A1 = B1/B10
  pmt =proc.time()[3]

  B1_py = r_to_py(B1, convert=FALSE); B2_py = r_to_py(B2, convert=FALSE); B10_py = r_to_py(B10, convert=FALSE);
  #y_py = r_to_py(y, convert=FALSE); X_py = r_to_py(X, convert=FALSE)
  gpu_ind = r_to_py(gpu_ind)
  lam_schd_py = r_to_py(lam_schd, convert = FALSE)
  if( is.null(type) == TRUE){
    type = fit_GMS[[2]][[1]]
  }
  print(type)
  fit = r_to_py(fit_GMS[[1]])
  if(is.null(y) == TRUE | is.null(X) == TRUE ){
    if( type == "CV"){
       print("X and y are missing!")
    }
  }
  custom = fit_GMS[[4]]
  custom_loss_file = custom[[1]]
  model = fit_GMS[[3]]
  if( is.null(custom_loss_file) == TRUE){
    if(model == "linear"){code_loss <- paste(system.file(package="GMS"), "Loss_Linear.py", sep="/")}
    if(model == "logistic"){code_loss <- paste(system.file(package="GMS"), "Loss_Logit.py", sep="/")}
    if(model == "LAD"){code_loss <- paste(system.file(package="GMS"), "Loss_LAD.py", sep="/")}
    #if(model == "cox"){code_loss <- paste(system.file(package="GMS"), "Loss_cox.py", sep="/")}
    if(model == "quantile"){code_loss <- paste(system.file(package="GMS"), "Loss_Quantile.py", sep="/")}
    reticulate::source_python(code_loss)#, envir = NULL,convert = FALSE)
  }else{
    model = NULL
    reticulate::source_python(custom_loss_file)#, envir = NULL,convert = FALSE)
  }

  type = r_to_py(type, convert = FALSE)
  code_Sampling <- paste(system.file(package="GMS"), "GMS_sampling_function.py", sep="/")
  reticulate::source_python(code_Sampling)#, envir = NULL,convert = FALSE)
  pmt = proc.time()[3]
  Theta1 = Theta2 = Theta_cv = Theta_stab = CV_err = NULL
  theta_hat  = NULL
  if(type == "CV"){
    samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
    Theta_cv = samples[[3]]
    CV_err = samples[[5]]
  }
  if(type == "NCV"){
    samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
    Theta_cv = samples[[3]]
    CV_err = samples[[5]]
  }
  if(type == "Boot-CV"){
    A1 = B1 / B10
    K = py_to_r(fit_GMS$type[[2]])
    CV_err_boot = array(0,dim=c(M,K,B1))
    for(m in 1:A1){
      ind = ((m-1)*B10+1):(m*B10)
      samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
      CV_err0 = samples[[5]]
      CV_err_boot[,,ind] = CV_err0
      pr = paste("",round(100*m/A1,1),"% completed")
      cat(pr, "\n")
    }
    type = "CV"
    samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
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
      samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
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
      samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
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
      samples = GMS_sampling(fit, lam_schd_py, y_py, X_py, B1_py, B2_py, B10_py, gpu_ind, type, eta)
      #samples = py_to_r(samples)
      Theta1[ind,] = samples[[1]]
      weight = samples[[8]]
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
    total = py_to_r(fit_GMS[[2]][[2]])*M*B1
  }

  if(type == "CV"){
    total = py_to_r(fit_GMS[[2]][[2]])*M
  }
  if(type == "NCV"){
    a = py_to_r(fit_GMS[[2]][[2]])
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
  out = list( Theta1 = Theta1, Theta2 = Theta2, Theta_cv = Theta_cv, CV_err = CV_err, Theta_stab = Theta_stab, lam_schd = lam_schd,
              time_samp = time_samp, theta_hat = theta_hat, type=type, eta = eta, n = n)
  if( type == "Boot-CV" ){
    out = list(Theta1 = Theta1, Theta2 = Theta2, Theta_cv = Theta_cv, CV_err_boot = CV_err_boot, CV_err0 = CV_err0, Theta_cv0 = Theta_cv0,
               lam_schd = lam_schd,time_samp = time_samp,  type=type, eta = eta, n =n)
  }
  if( type == "SingleBoot" ){
    out = list( Theta1 = Theta1, Theta2 = Theta2, Theta_cv = Theta_cv, CV_err = CV_err, Theta_stab = Theta_stab, lam_schd = lam_schd,
                time_samp = time_samp, theta_hat = theta_hat, type=type, eta = eta, n = n, weight = weight)
  }
  return( out )
}
