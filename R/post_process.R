#' #@importFrom Rcpp evalCpp
#' @title
#' Generative Bootstrap Sampler for Logisti Regression Model
#'
#' @description
#' Post-processing procedure to evaluate CIs and bias-corrected estimators.
#'
#' @param samples_GMS an output object from "GBS_Sampling" function.
#' @param alpha confidence level.
#' @param theta_hat the point estimator of the full data.
#' @param thre threshold to sparcify LASSO solutions. This value will be used to implement "StabSel".
#'
#' @usage
#' post_process(samples_GMS, alpha = 0.95, theta_hat = NULL,
#' thre = 0.005)
#'
#' @author Minsuk Shin, Jun Liu and Shijie Wang
#'
#' @seealso \code{\link{GMS_Sampling}}, \code{\link{GMS}}
#' @export
#' @examples
#' #post_process(samples_GBS, theta_hat = theta_hat, thre=0.001)
post_process <- function(samples_GMS, alpha = 0.95, theta_hat = NULL, thre = 0.005, type = NULL){
  require(reticulate)
  Theta1 = samples_GMS[[1]]
  Theta2 = samples_GMS[[2]]
  if( is.null(type) == TRUE ){
    type = samples_GMS[[9]]
  }
  Theta_cv = samples_GMS[[3]]
  CV_err = samples_GMS[[4]]
  Theta_stab = samples_GMS[[5]]
  lam_schd = samples_GMS[[6]]
  theta_cv_hat = NULL
  n = samples_GMS[[11]]
  if( is.null(theta_hat) == TRUE ){
     theta_hat = samples_GMS[[8]]
  }
  print("Post Process Evaluation Starts!")
  CI1 = CI2 = CI_cal = CI_perc = NULL
  Prop_stab = NULL
  lam_cv_hat = NULL
  if(type == "DoubleBoot"){
    B1 = dim(Theta2)[2]
    B2 = dim(Theta2)[1]
    p = dim(Theta2)[3]
    a0 = (1-alpha)/2
    T_s = matrix(0, B1, p)
    T0_s = matrix(0, B1, p)
    d = matrix(0, B1, p)
    sd0 = matrix(0, B1, p)
    #d0 = matrix(0, B1, p)
    CI1 = CI2 = CI_cal = matrix(0,p,2)
    CI_perc = matrix(0,p,2)
    pmt =proc.time()[3]
    cat(paste("Confidence Level: ", alpha,sep=""), fill=T)
    p0 = 1
    sd1 = apply(Theta1,2,sd)
    pmt = proc.time()[3]
    for(i in 1:B1){
      out = GMS::calib(Theta1[i,], Theta2[,i,], theta_hat, n, p, B1, B2, B1, B2)# GMS::calib(Theta1[i,], Theta2[,i,], theta_hat, n, p, B1, B2, B1, B2)
      T_s[i,] = out$T_s
      T0_s[i,] = out$T0_s
      d[i,] = as.vector(out$d)
      if( i %% round(B1/10) == 0 ){
        cat(paste(100*i/B1,"%"," | ",sep=""))
      }
    }
    for(j in 1:p){
      q_L = quantile(d[,j], a0)
      q_U = quantile(d[,j], 1-a0)
      if(q_L<0.0001) q_L=0.0001
      if(q_U<0.0001) q_U=0.0001
      if(q_L>0.9999) q_L=1-0.0001
      if(q_U>0.9999) q_U=1-0.0001
      CI2[j,1] = theta_hat[j] - quantile(T0_s[,j],1-a0)*sd1[j]
      CI2[j,2] = theta_hat[j] - quantile(T0_s[,j],a0)*sd1[j]
      CI_cal[j,1] = theta_hat[j] - quantile(Theta1[,j]-theta_hat[j], q_U)
      CI_cal[j,2] = theta_hat[j] - quantile(Theta1[,j]-theta_hat[j], q_L)
      CI1[j,1] = 2*theta_hat[j] - quantile(Theta1[,j],1-a0)
      CI1[j,2] = 2*theta_hat[j] - quantile(Theta1[,j],a0)
      CI_perc[j,1] = quantile(Theta1[,j], a0)
      CI_perc[j,2] = quantile(Theta1[,j], 1-a0)
    }
  }
  if(type == "SingleBoot"){
    B1 = dim(Theta1)[1]
    p = dim(Theta1)[2]
    a0 = (1-alpha)/2
    CI1  =  matrix(0,p,2)
    CI_perc = matrix(0,p,2)
    pmt =proc.time()[3]
    cat(paste("Confidence Level: ", alpha,sep=""), fill=T)
    pmt = proc.time()[3]
    for(j in 1:p){
      #CI2[j,1] = theta_hat[j] - quantile(T0_s[,j],1-a0)*sd1[j]
      #CI2[j,2] = theta_hat[j] - quantile(T0_s[,j],a0)*sd1[j]
      #CI_cal[j,1] = theta_hat[j] - quantile(T_s[,j],q_U)
      #CI_cal[j,2] = theta_hat[j] - quantile(T_s[,j],q_L)
      CI1[j,1] = 2*theta_hat[j] - quantile(Theta1[,j],1-a0)
      CI1[j,2] = 2*theta_hat[j] - quantile(Theta1[,j],a0)
      CI_perc[j,1] = quantile(Theta1[,j], a0)
      CI_perc[j,2] = quantile(Theta1[,j], 1-a0)
    }
  }

  if(type == "StabSel"){
    M = dim(Theta_stab)[1]
    B1 = dim(Theta_stab)[2]
    p = dim(Theta_stab)[3]
    Prop_stab = matrix(0,M,p)
    for(m in 1:M){
      theta = Theta_stab[m,,]
      prop = matrix(0,B1,p)
      ind = which(abs(theta)>thre)
      prop[ind] = 1
      Prop_stab[m,] = apply(prop,2,mean)
    }
  }
  if(type == "CV"){
    ind_cv_min = which.min(CV_err)
    lam_cv_hat = lam_schd[ind_cv_min]
    theta_cv_hat = apply(Theta_cv[,ind_cv_min,],2,mean)
  }
  if(type == "Boot-CV"){
    CV_err_boot = samples_GMS[[4]]
    B1 = dim(CV_err_boot)[3]
    K = dim(CV_err_boot)[2]
    CV_err0 = samples_GMS[[5]]
    Theta_cv0 = samples_GMS[[6]]
    lam_schd = samples_GMS[[7]]
    M = length(lam_schd)
    type = samples_GMS[[9]]
    lam_cv_boot = rep(0,B1)
    CV_err_boot_min = rep(0,B1)
    CV_err_boot_lam = matrix(0,M,B1)
    for(i in 1:B1){
      CV_err1 = CV_err_boot[,,i]
      err = apply(CV_err1,1,mean)
      id_min = which.min(err)
      lam_hat = lam_schd[ id_min ]
      lam_cv_boot[i] = lam_hat
      CV_err_boot_min[i] = min(err)
      CV_err_boot_lam[ , i] = err
      if( i %% round(B1/20) == 0){
        pr = paste("",round(100*i/B1,1),"% completed")
        cat(pr, "\n")
      }
    }
  }
  cat("Done!", fill=T)
  out = NULL
  if( type == "DoubleBoot"){
    out = list(type=type, CI_simple = CI1, CI_student = CI2, CI_cal = CI_cal, CI_perc = CI_perc, alpha=alpha)
  }
  if( type == "SingleBoot" ){
    out = list(type=type, CI_simple = CI1, CI_perc = CI_perc, alpha=alpha)
  }
  if( type == "CV"| type == "Tuning" ){
    out = list(type=type, lam_cv_hat = lam_cv_hat, CV_err = CV_err, theta_cv_hat = theta_cv_hat,
               lam_schd = as.vector(lam_schd))
  }
  if( type == "NCV" ){
    CV_err_hat = mean(CV_err)
    out = list(type = type, CV_err_hat = CV_err_hat, CV_err = CV_err, lam_hat = Theta_cv)
  }
  if( type == "StabSel"){
    out = list(type=type, Prop_stab = Prop_stab, lam_schd = as.vector(lam_schd))
  }
  if(type == "Boot-CV"){
    out = list(type= type, lam_hat_boot = lam_cv_boot,
               CV_boot_min = CV_err_boot_min, CV_boot_lam = CV_err_boot_lam,
               Theta_cv0 = Theta_cv0,  CV_err0 = CV_err0, lam_schd = as.vector(lam_schd))
  }

  return(out)
}





