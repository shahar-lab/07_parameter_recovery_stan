data {
  int<lower = 0> nTrials; //number of trials
  int<lower = 2> nArms; //number of alternatives 
  int<lower = 0> a1[nTrials]; //index of which arm was pulled
  int<lower = 0> reward[nTrials]; //outcome of bandit arm pull
}

parameters {
  real alpha; //learning rate
  real beta; //softmax parameter - inverse temperature
}

transformed parameters {
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
    delta[trial] = reward[trial] - Q[trial, a1[trial]];

    //update Q value based on prediction error (delta) and learning rate (alpha)
    Q[trial, a1[trial]] = Q[trial, a1[trial]] + inv_logit(alpha) * delta[trial];
  }
}

model {
  // priors
  beta ~ normal(0, 1); 
  alpha ~ normal(0, 1);

  for (trial in 1:nTrials) {
    //returns the probability of having made the choice you made, given your beta and your Q's
    target += log_softmax(Q[trial] *exp(beta))[a1[trial]];
  }
}
