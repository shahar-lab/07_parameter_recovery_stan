data {
  // nY: number of trials total
  int<lower=1> nY ;
  // nX: number of within-subject predictors in model matrix X
  int<lower=1> nX ;
  // nS: number of subjects
  int<lower=1> nS ; 
  // S: subject labels for each trial
  int<lower=1,upper=nS> S[nY] ; 
  // X: model matrix of within-subject predictors
  matrix[nY,nX] X ; 
  // Y: trial outcomes
  vector[nY] Y ; 
}
parameters {
  // sigma: trial-by-trial error
  real<lower=0> sigma ; 
  // mu: population-level means for all effects for predictors in model matrix X
  row_vector[nX] mu ; 
  // tau: SDs of subjects' deviations from all population level effects in mu
  vector<lower=0>[nX] tau ; 
  // L_Omega: used for evaluating correlation amongst subjects' deviations
  cholesky_factor_corr[nX] L_Omega ;
  // betas01: intermediate variable, used for faster non-centered parameterization 
  matrix[nX,nS] betas01 ; 
}
transformed parameters {
  // Smu: subject-by-subject values for each effect
  matrix[nS,nX] Smu ;
  
  // compute deviations  
  Smu = (diag_pre_multiply(tau,L_Omega)*betas01)' ;
  // add mu to deviations
  for(ns in 1:nS){
    Smu[ns] = Smu[ns] + mu ;
  }
}
model {
  //priors on population parameters
  sigma ~ normal(0,1) ; //left-bounded at zero thanks to declaration above
  mu ~ normal(0,1) ;
  tau ~ normal(0,1) ; 
  L_Omega ~ lkj_corr_cholesky(2) ; //slightly higher mass at zero than uniform

  // assert sampling of betas01 as standard normal
  to_vector(betas01) ~ normal(0,1) ;

  //assert sampling of trial-by-trial data 
  Y ~ normal( rows_dot_product(Smu[S], X) , sigma ) ;
}
