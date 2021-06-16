data {
int<lower=1> N; // number of observations
  int<lower=1> J; // dimension of observations
  vector[J] y[N]; // observations
  vector[J] Zero; // a vector of Zeros (fixed means of observations)
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
corr_matrix[J] Omega; 
  vector<lower=0>[J] sigma; 
}
transformed parameters {
  cov_matrix[J] Sigma; 
  Sigma = quad_form_diag(Omega, sigma);
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
y ~ multi_normal(Zero,Sigma); // sampling distribution of the observations
  sigma ~ cauchy(0, 5); // prior on the standard deviations
  Omega ~ lkj_corr(1); // LKJ prior on the correlation matrix 
  }

