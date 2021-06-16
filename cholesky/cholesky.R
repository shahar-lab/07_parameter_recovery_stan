library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

set.seed(666)
Omega <- rbind(
  c(1, 0.3, 0.2),
  c(0.3, 1, 0.1),
  c(0.2, 0.1, 1)
)
sigma <- c(1, 2, 3)
Sigma <- diag(sigma) %*% Omega %*% diag(sigma)
N <- 100
y <- mvtnorm::rmvnorm(N, c(0,0,0), Sigma)



stanmodel1 <- stan_model(model_code = stancode1, model_name="stanmodel1")
standata <- list(J = ncol(y), N=N, y = y, Zero=rep(0, ncol(y)))
estimates <- function(y, perturb=FALSE){
  if(perturb) y <- y + rnorm(length(y), 0, 1)
  sigma <- sqrt(diag(var(y)))
  Omega <- cor(y)
  return(list(sigma=sigma, Omega=Omega))
}
inits <- function(chain_id){
  values <- estimates(standata$y, perturb = chain_id > 1)
  return(values)
}

samples1 <- sampling(stanmodel1, data = standata, 
                     iter = 2000, warmup = 1000, chains = 4)

library(coda)
codasamples1 <- do.call(mcmc.list, 
                        plyr::alply(rstan::extract(samples1, 
                                                   pars=c("sigma", "Omega[1,2]", "Omega[1,3]", "Omega[2,3]"), 
                                                   permuted=FALSE), 2, mcmc))
