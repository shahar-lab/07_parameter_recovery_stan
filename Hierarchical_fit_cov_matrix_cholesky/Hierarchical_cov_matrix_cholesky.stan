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
  vector[Nparam] mu;
  vector<lower=0>[Nparam] tau; //vector of random effects variance
  cholesky_factor_corr[Nparam] L_Omega;

}

transformed parameters {
      real alpha[Nsubj];
      real beta[Nsubj];
      matrix [Nsubj,Ntrials] log_lik;
      vector<lower=0, upper=1>[Narms] Q;
      
      matrix[Nparam,Nparam] sigma_matrix;
      sigma_matrix = diag_pre_multiply(tau, (L_Omega*L_Omega')); //L_Omega*L_omega' give us Omega (the corr matrix). 
      sigma_matrix = diag_post_multiply(sigma_matrix, tau);     // diag(tau)*omega*diag(tau) gives us sigma_matirx (the cov matrix)


  for (subj in 1:Nsubj){
        real PE;
        alpha[subj] = inv_logit(auxiliary_parameters[subj][1]); 
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
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 1);             //tau is the hyperparameters variance vector
  L_Omega ~ lkj_corr_cholesky(2); //L_omega is the lower triangle of the correlations. Setting the lkj prior to 2 means the off-diagonals are priored to be near zero


  // indvidual level priors (subject parameters)
  auxiliary_parameters ~ multi_normal(mu, sigma_matrix);

  target += sum(log_lik);

}

