data {
  int<lower = 1> Nsubj; //number of subjects
  int<lower = 1> Ntrials; //number of trials
  int<lower = 2> Narms; //number of alternatives 
  int<lower = 1> Nparam; //number of parameters
  int<lower = 0> a1[Nsubj,Ntrials]; //index of which arm was pulled
  int<lower = 0, upper = 1> reward[Nsubj,Ntrials]; //outcome of bandit arm pull
}

parameters {
  vector[Nparam] auxiliary_parameters[Nsubj]; 

  //hyper parameters 
  real alpha_aux_mu;
  //real <lower = 0> alpha_sigma;
  real beta_aux_mu;
  //real <lower = 0> beta_sigma;
  //real <lower = 0> tau_alpha_beta; 
  vector<lower=0>[Nparam] tau; //diagnoal sd of the corr matrix
  cholesky_factor_corr[Nparam] L_Omega;

}

transformed parameters {
      vector[Nparam] mu;
      real alpha[Nsubj];
      real beta[Nsubj];
      matrix [Nsubj,Ntrials] log_lik;
      vector<lower=0, upper=1>[Narms] Q;
      
      matrix[Nparam,Nparam] sigma_matrix;

      sigma_matrix = diag_pre_multiply(tau, L_Omega);
      sigma_matrix *= sigma_matrix';
      mu[1] = alpha_aux_mu;
      mu[2] = beta_aux_mu;


  for (subj in 1:Nsubj){
        real PE;
        alpha[subj] = inv_logit(auxiliary_parameters[subj][1]); //[subj,1]
        beta[subj] = exp(auxiliary_parameters[subj][2]);


        for (a in 1:Narms) Q[a] = 0;
        
        for (trial in 1:Ntrials){
            log_lik[subj,trial]=log_softmax(Q*beta[subj])[a1[subj,trial]];
            PE= reward[subj,trial] - Q[a1[subj,trial]];
            Q[a1[subj,trial]] += alpha[subj] * PE;
        } 
  }

}
model {
  
  // population level priors (hyper-parameters)
  alpha_aux_mu ~ normal(0, 5);
  beta_aux_mu ~ normal(0, 5);
 // tau_alpha_beta ~ normal(0, 5);
  tau ~ cauchy(0, 1);
  L_Omega ~ lkj_corr_cholesky(2);



  // indvidual level priors (subject parameters)
  auxiliary_parameters ~ multi_normal(mu, sigma_matrix);

  /*for (subj in 1:Nsubj){
  auxiliary_parameters[subj] ~ multi_normal(mu, sigma_matrix);
  }*/

  target += sum(log_lik);

}

