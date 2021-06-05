data {
  int<lower = 1> Nsubj; //number of subjects
  int<lower = 1> Ntrials; //number of trials
  int<lower = 2> Narms; //number of alternatives 
  int<lower = 0, upper = 2> a1[Nsubj,Ntrials]; //index of which arm was pulled
  int<lower = 0, upper = 1> reward[Nsubj,Ntrials]; //outcome of bandit arm pull
  int final_trl[1,Nsubj]; //number of trials (with no NA) 

}

parameters {
  real alpha_subjects[Nsubj]; //learning rate
  real beta_subjects[Nsubj]; //softmax parameter - inverse temperature

  //hyper parameters 
  real mu_alpha;
  real <lower = 0> tau_alpha;
  real mu_beta;
  real <lower = 0> tau_beta;
}

transformed parameters {
      real alpha[Nsubj];
      real beta[Nsubj];
      matrix [Nsubj,Ntrials] log_lik;
      vector<lower=0, upper=1>[Narms] Q;
      log_lik = rep_matrix(0,Nsubj,Ntrials);


  for (subj in 1:Nsubj){
        real PE;
        alpha[subj] = inv_logit(alpha_subjects[subj]);
        beta[subj] = exp(beta_subjects[subj]);

        for (a in 1:Narms) Q[a] = 0;
        
        for (trial in 1:final_trl[1,subj]){
            log_lik[subj,trial]=log_softmax(Q*beta[subj])[a1[subj,trial]];
            PE= reward[subj,trial] - Q[a1[subj,trial]];
            Q[a1[subj,trial]] += alpha[subj] * PE;
        } 
  }
}
model {
  
  // population level priors (hyper-parameters)
  mu_alpha ~ normal(0, 5);
  tau_alpha ~ normal(0, 5);
  mu_beta ~ normal(0, 5);
  tau_beta ~ normal(0, 5);

  // indvidual level priors (subject parameters)
  beta_subjects ~ normal(mu_beta, tau_beta);
  alpha_subjects ~ normal(mu_alpha, tau_alpha);

  target += sum(log_lik);

}

