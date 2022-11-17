//[[Rcpp::depends( RcppArmadillo )]]
#include <RcppArmadillo.h>
//#include <RcppArmadilloExtensions/sample.h>
#include <Rcpp/Benchmark/Timer.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List calib_new(arma::vec theta1, arma::mat theta2, arma::vec theta, int n,  int p,
                         int B1, int B2, double B10, double B20){
  arma::vec T_s(p);
  arma::vec T0_s(p);
  arma::mat T_ss(B2,p);
  //arma::mat T0_ss(B2,p);
  arma::mat sd2(1,p);
//arma::mat sd3(1,p);
  arma::mat d(1,p);
  //arma::mat d0(1,p);
  //int i=0;
//  int j=0;
  int u=0;
  int k=0;
  int k1=0;
  double mu=0.0;
  //Rcpp::Rcout << 1 << std::endl;
  for(u=0; u<p; u++){
    d[u] = 0.0;
//    d0[u] = 0.0;
  }
  //Rcpp::Rcout << 1 << std::endl;
  for(u=0; u<p; u++){
    mu = 0.0;
    for(k=0; k<B2; k++){
      mu += theta2(k,u)/B20;
    }
    //Rcpp::Rcout << mu << std::endl;
    sd2(0,u) = 0.0;
    for(k=0; k<B2; k++){
      sd2(0,u) += (theta2(k,u)-mu)*(theta2(k,u)-mu)/(B20-1.0);
    }
    sd2(0,u) = sqrt(sd2(0,u));
    T_s[u] = 2*theta[u] - theta1[u];
    T0_s[u] = (theta1[u] - theta[u])/sd2(0,u);
    //Rcpp::Rcout << sd2 << std::endl;
    for(k=0; k<B2; k++){
      //mu = 0.0;
      //for(k1=0; k1<B3; k1++){
      //  mu += theta3(k1,k,u)/B30;
      //}
      //sd3(0,u) = 0.0;
      //for(k1=0; k1<B3; k1++){
      //  sd3(0,u) += (theta3(k1,k,u)-mu)*(theta3(k1,k,u)-mu)/(B30-1.0);
      //}
      //sd3(0,u) = sqrt(sd3(0,u));
      T_ss(k,u) = 2*theta1[k] - theta2(k,u);
      //T0_ss(k,u) = (theta2(k,u) - theta1[u])/sd3(0,u);
      if(T_ss(k,u) < T_s[u]){
        d(u)+= 1.0/B20;
      }
      //if(T0_ss(k,u) < T0_s[u]){
      //  d0(u)+= 1.0/B20;
      //}
    }
  }



  return Rcpp::List::create(Rcpp::Named("T_s") =T_s,
                            Rcpp::Named("T0_s") =T0_s,
                            //Rcpp::Named("sd1") = sd1,
                            Rcpp::Named("sd2") = sd2,
//                            Rcpp::Named("sd3") = sd3,
                            //Rcpp::Named("mu") = mu,
                            Rcpp::Named("d") = d
                            //Rcpp::Named("d0") = d0
  );
}



