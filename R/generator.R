#' @title
#' Generator of GBS for a given weight
#' @description
#' Generate the weighted M-estimator for a given weight and a tuning parameter.
#' @usage
#' generator(fit_GBS, w, lam, eta = NULL)
#' @param fit_GMS the trained GMS object.
#' @param w weights: a B X S matrix.
#' @param lam tuning parameter: a scalar.
#' @param eta auxiliary parameter: a scalar.
#'
#' @export
#' @seealso \code{\link{GMS_Sampling}}, \code{\link{GMS}}, \code{\link{post_process}}, \code{\link{GMS_Loading}}
#' @author Minsuk Shin, Jun Liu and Shijie Wang
#' @examples
#' theta_hat = generator(fit_GMS, w=matrix(1,1,S), verb=0)
generator <- function(fit_GMS, w, lam = 1, eta = NULL, eta_cand = NULL, verb = 1){
  if( is.vector(w) == TRUE){
    w = matrix(w,1,length(w))
  }
  if( is.null(eta) == TRUE ){
    eta = 0.5
  }

  if( is.null(eta_cand) == TRUE ){
    eta = eta*matrix(1,dim(w)[1],1)
  }else{
    if( is.vector(eta_cand) == TRUE ){
      eta_cand = matrix(eta_cand, length(eta_cand), 1)
    }
    eta = eta_cand
  }

  lam1 = lam*matrix(1,dim(w)[1],1)
  B = dim(w)[1]
  a = paste("lam: ", round(lam,7), sep="")
  b = paste("Total ",B, " evaluations starts!", sep="")
  if(verb ==  1){
    print(a)
    print(b)
  }
  fit = r_to_py(fit_GMS[[1]])
  S = py_to_r(fit[9])
  if( S != dim(w)[2] ){
    print(paste("The value of S is not matched! S is supposed to be",S))
  }
  code_Sampling <- paste(system.file(package="GMS"), "GMS_generator_function.py", sep="/")
  reticulate::source_python(code_Sampling)#, envir = NULL,convert = FALSE)

  w = r_to_py(w, convert = FALSE)
  lam1 = r_to_py(lam1, convert = FALSE)
  eta = r_to_py(eta, convert = FALSE)
  Theta = GMS_generator(fit, w, lam1, eta)
  #Theta = py_to_r(samples)
  return( Theta )
}

