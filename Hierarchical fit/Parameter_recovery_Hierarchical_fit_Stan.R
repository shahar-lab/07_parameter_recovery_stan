#aim: perform data simulation for multi-armed bandit and recover with hierarchical modeling (with no correlation between random effects)
#contributor: Shira Niv, Nitzan Shahar

rm(list=ls())

library("rstan") 
library("truncnorm")
library(parallel)
library(gtools) #inv.logit function 

source('sim_Narmed_bandit.R')
rndwlk<-read.csv('rndwlk_4frc_1000trials.csv',header=F)


# simulate data -----------------------------------------------------------
#generate parameters and data for N agents. 
Nsubj   <-100          #number of agents
Nalt<-4               #number of alternatives
Ntrl<-200           #number of trials

alpha_mu = 0.5 
beta_mu = 3
alpha_sigma = 0.2 
beta_sigma = 2

true.parms<-data.frame(alpha=rtruncnorm(Nsubj, mean =alpha_mu, sd = alpha_sigma, a = 0,b = 1),beta =rtruncnorm(Nsubj, mean =beta_mu, sd = beta_sigma, a = 0.01,b = 10))
hist(true.parms$alpha)
hist(true.parms$beta)
mean(true.parms$alpha)
sd(true.parms$alpha)
mean(true.parms$beta)
sd(true.parms$beta)

df<-  mclapply(1:Nsubj,function(s)   {
                      sim.block(Ntrl,Nalt,true.parms$alpha[s],true.parms$beta[s],rndwlk)},
                      mc.cores=4)


# parameter recovery with stan --------------------------------------------
str(df) #checking the data frame 
start_time <- Sys.time()


      #prepare action and reward matrices (subject x trial)
      a1=t(sapply(1:Nsubj,function(subj) {df[[subj]][,'action']}))
      reward=t(sapply(1:Nsubj,function(subj) {df[[subj]][,'reward']}))
      
      #fit stan model
      model_data <- list(Nsubj = Nsubj,
                         Ntrials = Ntrl,
                         Narms = Nalt,
                         a1 = a1,
                         reward = reward)
        
rl_fit<- stan(file = "rl_Hierarchical_1.stan", data=model_data, iter=2000,chains=4,cores =4) #iter - number of MCMC samples 

print(rl_fit)
save(rl_fit,"rl_fit.Rdata")
save(rl_fit,file='rl_fit.Rdata')


library("shinystan")
launch_shinystan(rl_fit)
end_time <- Sys.time()

#  compare parameters --------------------------------------------

#population level (hyperparameter)
alpha_population_recovered=summary(rl_fit , pars=c("mu_alpha"))$summary[,1]
beta_population_recovered=summary(rl_fit , pars=c("mu_beta"))$summary[,1]
alpha_tau_population_recovered=summary(rl_fit , pars=c("tau_alpha"))$summary[,1]
beta_tau_population_recovered=summary(rl_fit , pars=c("tau_beta"))$summary[,1]

inv.logit(alpha_population_recovered)
exp(beta_population_recovered)
inv.logit(alpha_tau_population_recovered)
exp(beta_tau_population_recovered)

#individual level parameters (subjects parameters)
alpha_recovered=summary(rl_fit , pars=c("alpha_subjects"))$summary[,1] 
beta_recovered=summary(rl_fit , pars=c("beta_subjects"))$summary[,1]
plot(true.parms[,1],inv.logit(alpha_recovered))
plot(true.parms[,2],exp(beta_recovered))

cor(true.parms[,1],inv.logit(alpha_recovered))
cor(true.parms[,2],exp(beta_recovered))






# Arch --------------------------------------------------------------------

#options(mc.cores = parallel::detectCores())
#rstan_options(auto_write=TRUE)
#2nd version
options(mc.cores = parallel::detectCores())
rstan_options(auto_write=TRUE)

my_model<- stan_model(file = "rl_basic.stan") 
sample <- sampling(object = my_model, data = model_data)

fit <-optimizing(object = my_model, data = model_data)
#c(fit$par[1],fit$par[2])



my_model<- stan_model(file = "rl_basic.stan") 
sample <- sampling(object = my_model, data = model_data)

plot(sample, plotfun = "hist", pars= "alpha")
plot(sample, plotfun = "hist", pars= "beta")

library("shinystan")
launch_shinystan(sample)

#calculate cor between true and recovered params
df.tbl   <-lapply(1:length(Nalt), function(alt) {
  lapply(1:length(Ntrl), function(trl) {
    data.frame(Nalt=Nalt[alt],
               Ntrl=Ntrl[trl],
               cor.alpha=cor(true.parms$alpha,inv.logit((do.call(rbind,alpha[[alt]][[trl]])))),
               cor.beta=cor(true.parms$beta,exp((do.call(rbind,beta[[alt]][[trl]])))))
  })})

df.tbl<-do.call(rbind,lapply(1:length(Nalt), function(alt) {do.call(rbind,df.tbl[[alt]])}))

require(dplyr)
library(tidyr)
library("tidyverse")
library(knitr) #for printing the correlation table to a file
library(kableExtra) #for printing the correlation table to a file
library(triangle) #triangular distribution
library(msm)#for generating the parameters 


my_model <- stan_model(file = "rl_basic.stan")
fit <- optimizing(object = my_model, data = model_data)

#get alpha and beta estimates
fit$par[1]
fit$par[2]


#if Rhat is not close to 1.01 it means the model ran efficiently 
#comparing to glm
summary(glm(stay1_bandit~reward_oneback, family = "binomial", data = df)) #should induce similar estimates values
#plot 
traceplot(log_fit,c("alpha","beta"),ncol=1,nrow=6,inc_warmup=F)





#

#print table to file
df.tbl %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))        


