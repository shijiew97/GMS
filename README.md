# GMS
This is an R package to implement the Generative Multiple-purpose Sampler (GMS) model in the paper, "Generative Multiple-purpose Sampler for Weighted M-estimation" by Minsuk Shin, Shijie Wang and Jun S Liu. [[paper]](https://arxiv.org/abs/2006.00767)

__GMS is a general computational framework to accelerate repteted calculations for (penalized) weighted M-estimations including (Single/Double) bootstrap methods, cross-validation and non-parametric empirical Bayes. A various of models are exmained in paper such as LASSO, quantile regression, logistics regression and nonparametric maximum likelihodd estiamtion (NPMLE).__

### 1. Installing the Package
In order to sucessfully run the GMS model, there are several pre-requisites need to be installed before the R package. The main of GMS is in `Python`, especially __Pytorch__ library and we strongly recommend using `CUDA` (GPU-Based tool) to train GMS which can be accelerated a lot than using `CPU`.
- __Python__ 3.7 or above
- __[Pytroch](https://pytorch.org/)__ 1.11.0 or above
- __[NAVID CUDA](https://developer.nvidia.com/cuda-toolkit)__ 10.2 or above

In R, we also need `reticulate` package to run `Python` in R and `devtools` to install R package from github.
```
install.package("reticulate")
install.package("devtools")
```

Now, use the following code to install __GMS__ package.
```
library(devtools)
install_github(repo = "Jackie97zz/GMS")
library(GMS)
```

### 2. Main function
There are three main functions in the `GMS` package, which is detailed specified below.
- `GMS` which aims to train the generator of GMS. It takes severals inputs 





