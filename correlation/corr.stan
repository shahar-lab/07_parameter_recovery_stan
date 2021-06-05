data {
  int<lower = 1> Nsubj; 
  vector[Nsubj] x;
  vector[Nsubj] y;
}

parameters {
  real r;
  real<lower=0> sigma;
}

model {
  r~normal(0.9,0.1);
  y ~ normal(r * x, sigma);
}

