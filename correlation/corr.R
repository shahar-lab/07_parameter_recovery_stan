# Parameter recovery for correlation coefficient 
rm(list=ls())

library("rstan") # observe startup messages
library(parallel)
library(MASS)

# simulate data -----------------------------------------------------------
#generate parameter and data for N agents. 
Nsubj  =50          #number of agents
True_r = 0

df=mvrnorm(Nsubj,mu=c(0,0),Sigma=matrix(c(1,True_r,True_r,1),2))
cor(df)

#fit stan model
model_data <- list(x = df[,1],
                   y = df[,2],
                   Nsubj=Nsubj)

fit<- stan(file = "corr.stan", data=model_data, iter=2000,chains=4,cores =4) #iter - number of MCMC samples 

library("bayesplot")
posterior <- as.matrix(fit)

plot_title <- ggtitle("Posterior distributions",
                      "with medians and 80% intervals")
mcmc_areas(posterior,
           pars = c("r","sigma"),
           prob = 0.95) + plot_title


library("shinystan")
launch_shinystan(fit)
