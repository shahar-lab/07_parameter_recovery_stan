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
  for (subj in 1:Nsubj){
    alpha[subj] = inv_logit(alpha_subjects[subj]);
    beta[subj] = exp(beta_subjects[subj]);
    vector<lower=0, upper=1>[nArms] Q[nTrials];  // value function for each arm
    real delta[nTrials];  // prediction error

       for (trial in 1:nTrials) {

       //set initial Q and delta for each trial
         if (trial == 1) {
        //if first trial, initialize Q values as specified
        for (a in 1:nArms) {
        Q[1, a] = 0;
        }
        } else {
        //otherwise, carry forward Q from last trial to serve as initial value
       for (a in 1:nArms) {
        Q[trial, a] = Q[trial - 1, a];
      }
    }

    //calculate prediction error and update Q (based on specified beta)
    delta[trial] = reward[subj,trial] - Q[trial, a1[subj,trial]];

    //save likelihood
    
    like=log_softmax(Q[trial] * beta[subj])[a1[subj,trial]]
    
    
    //update Q value based on prediction error (delta) and learning rate (alpha)
    Q[trial, a1[subj,trial]] = Q[trial, a1[subj,trial]] + alpha * delta[trial];
  }
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

