% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/GMS_Sampling.R
\name{GMS_Sampling}
\alias{GMS_Sampling}
\title{Sampling Function of Generative Bootstrap Sampler}
\usage{
GMS_Sampling(fit_GMS, B1=3000, B2=100, B10=NULL, lam_schd = NULL,
X=NULL, y=NULL, gpu_ind = 0, type = NULL, eta = NULL)
}
\arguments{
\item{fit_GMS}{an GMS object evaluated by "GMS" function.}

\item{B1}{the number of bootstrap samples in the first level.}

\item{B2}{the number of bootstrap samples in the second level.}

\item{B10}{the number of the first level bootstrap samples to be
evaluated at each iteration. This is to reduce the amount of RAM and
GPU memory in storing the bootstrap samples.}

\item{lam_schd}{a candidate set of tuning parameter for CV and
stability selection. When the value is not specified, the candidate set
stored in "fit_GBS" will be used.}

\item{X}{the predictor used in training the generator.
This should be given to evaluate a CV.}

\item{y}{the response used in training the generator.
This should be given to evaluate a CV.}

\item{gpu_ind}{gpu index to run the computation. Useful under multiple GPU environment.}
}
\description{
Generate bootstrap sample from the trained generator of GBS.
}
\examples{
#samples_GBS = GBS_Sampling(fit_GBS, X = X, y = y)
}
\seealso{
\code{\link{post_process}}, \code{\link{GMS}}, \code{\link{GMS_Loading}}, \code{\link{generator}}
}
\author{
Minsuk Shin, Jun Liu and Shijie Wang
}
