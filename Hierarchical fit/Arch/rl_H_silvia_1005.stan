functions{
  real[] fit(int nTrials, int nArms, int[] a1,int[] reward, real alpha, real beta){
      real log_lik[nTrials];
      real PE;
      vector<lower=0, upper=1>Q[nArms];


      // Initializing values
        for (a in 1:nArms) {
          Q[a] = 0;
          }
        
        for (trial in 1:nTrials){
            log_lik[trial] = 0;
            log_lik[trial]+=log_softmax(Q)[a1[trial]];
            PE= reward[trial] - Q[a1[trial]];
            Q[a1[trial]] += alpha * PE;
            
        } 
       return log_lik;

} 
  


data {
  int<lower = 0> Nsubj; //number of subjects
  int<lower = 0> nTrials; //number of trials
  int<lower = 2> nArms; //number of alternatives 
  int<lower = 1, upper = 2> a1[Nsubj,nTrials]; //index of which arm was pulled
  int<lower = 0, upper = 1> reward[Nsubj,nTrials]; //outcome of bandit arm pull
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
      vector<lower=0, upper=1>[nArms] Q[nTrials];
      real alpha[Nsubj];
      real beta[Nsubj];
      real log_lik[Nsubj,nTrials];

  for (subj in 1:Nsubj){
    alpha[subj] = inv_logit(alpha_subjects[subj]);
    beta[subj] = exp(beta_subjects[subj]);
  }
  
      for (subj in 1:Nsubj){
        log_lik[subj] = fit(Nsubj, a1[subj], reward[subj],alpha[subj], beta1[subj]);
      }
            
}
model {
   // hyper priors
  mu_alpha ~ normal(0, 5);
  tau_alpha ~ normal(0, 5);
  mu_beta ~ normal(0, 5);
  tau_beta ~ normal(0, 5);

  // priors
  beta_subjects ~ normal(mu_beta, tau_beta);
  alpha_subjects ~ normal(mu_alpha, tau_alpha);


  for (subj in 1:Nsubj) {
       //returns the probability of having made the choice you made, given your beta and your Q's
    target += sum(like[subj]);
  }
}

