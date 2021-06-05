data {
  int<lower = 1> Nsubj; //number of subjects
  int<lower = 1> Ntrials; //number of trials
  int<lower = 2> Narms; //number of alternatives 
  int<lower = 1> Nparam; //number of parameters 
  int<lower = 0, upper = 2> a1[Nsubj,Ntrials]; //index of which arm was pulled
  int<lower = 0, upper = 1> reward[Nsubj,Ntrials]; //outcome of bandit arm pull
}

parameters {
  real alpha_subjects[Nsubj]; //learning rate
  real beta_subjects[Nsubj]; //softmax parameter - inverse temperature

  //hyper parameters 
  real mu_alpha;
  real <lower = 0> tau_alpha;
  real mu_beta;
  real <lower = 0> tau_beta;
  
   // Transformed model parameters
    vector[Nparam] subjparams[N];
    cholesky_factor_corr[Nparam] L_Omega;
    vector<lower=0>[Nparam] tau;
    vector[Nparam] mu;
  
}

transformed parameters {
      real alpha[Nsubj];
      real beta[Nsubj];
      matrix[Nparam, Nparam] Sigma;
      matrix [Nsubj,Ntrials] log_lik;
      vector<lower=0, upper=1>[Narms] Q;
      log_lik = rep_matrix(0,Nsubj,Ntrials);
       Sigma = diag_pre_multiply(tau, L_Omega);
       Sigma *= Sigma';


  for (subj in 1:Nsubj){
        real PE;
        alpha1[i] = inv_logit(subjparams[i][1]);
        beta1[i] = exp(subjparams[i][2]);

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
  tau ~ cauchy(0, 1);
  L_Omega ~ lkj_corr_cholesky(2);

  subjparams ~ multi_normal(mu, Sigma);


  target += sum(log_lik);

}

