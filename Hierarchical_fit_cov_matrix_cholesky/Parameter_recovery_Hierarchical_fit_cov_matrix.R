# Parameter recovery Hierarchical fit Stan - covariance matrix 

rm(list=ls())

library('rstan') # observe startup messages
library("truncnorm")
library(parallel)
library(gtools) #inv.logit function 
library(MASS)

source('sim_Narmed_bandit.R')
rndwlk<-read.csv('rndwlk_4frc_1000trials.csv',header=F)


# simulate data -----------------------------------------------------------
#generate parameters and data for N agents. 
Nsubj   <-50         #number of agents
Nalt<-4             #number of alternatives
Ntrl<-200           #number of trials
Nparam <- 2         #number of parameters

#population parameters
alpha_aux_mu = logit(0.5)
beta_aux_mu = log(5)
alpha_sigma = 0.5
beta_sigma = 0.2
tau_alpha_beta = 0.6

#covariance matrix
mu_vector=c(alpha_aux_mu,beta_aux_mu)
sigma_matrix = matrix(
  data = c(
    alpha_sigma^2
    , alpha_sigma*beta_sigma*tau_alpha_beta
    , alpha_sigma*beta_sigma*tau_alpha_beta
    , beta_sigma^2
  )
  , nrow = 2
  , ncol = 2
)

#actually sample the population
auxiliary_parameters = mvrnorm(n = Nsubj, mu = mu_vector, Sigma = sigma_matrix)


#check that we got data with statistics as expected
mean(auxiliary_parameters[,1])
inv.logit(mean(auxiliary_parameters[,1]))
mean(auxiliary_parameters[,2])
exp(mean(auxiliary_parameters[,2]))
sd(auxiliary_parameters[,1])
sd(auxiliary_parameters[,2])
cor(auxiliary_parameters[,1],auxiliary_parameters[,2])

true.parms <-auxiliary_parameters
colnames(true.parms)<-c("alpha","beta")

true.parms[,1]<-inv.logit(true.parms[,1])
true.parms[,2]<-exp(true.parms[,2])

hist(true.parms[,1])
hist(true.parms[,2])


# simulating N agents in the 2 step task 

df<- lapply(1:Nsubj,function(s)           {
                      
                      df_subj=cbind(subj=rep(s,Ntrl),
                                    trial=(1:Ntrl),
                                    sim.block(Ntrl,Nalt,true.parms[s,1],true.parms[s,2],rndwlk))
                      })

df<-do.call(rbind,df)


# parameter recovery with stan --------------------------------------------

#prepare action and reward matrices (subject x trial)
a1=t(sapply(1:Nsubj,function(subj) {df[df$subj==subj,'action']}))
reward=t(sapply(1:Nsubj,function(subj) {df[df$subj==subj,'reward']}))

#fit stan model
      model_data <- list(Nsubj = Nsubj,
                         Ntrials = Ntrl,
                         Narms = Nalt,
                         a1 = a1,
                         reward = reward,
                         Nparam = Nparam
                         )
        
        #options(mc.cores = parallel::detectCores())
        #rstan_options(auto_write=TRUE)
        
        rl_fit<- stan(file = "Hierarchical_cov_matrix_basic.stan", data=model_data, iter=2000,chains=4,cores =4) #iter - number of MCMC samples 

        #m = summary(rl_fit , pars=c("alpha","beta"))
        #c(m$summary[1,1],m$summary[2,1])


###
        print(rl_fit)
        
        #population level (hyperparameter)
        alpha_aux_mu_recovered = (summary(rl_fit , pars=c("alpha_aux_mu"))$summary[,1])
        beta_aux_mu_recovered = summary(rl_fit , pars=c("beta_aux_mu"))$summary[,1]
        alpha_sigma_recovered = summary(rl_fit , pars=c("alpha_sigma"))$summary[,1]
        beta_sigma_recovered = summary(rl_fit , pars=c("beta_sigma"))$summary[,1]
        tau_alpha_beta_recovered = summary(rl_fit , pars=c("tau_alpha_beta"))$summary[,1]
        
        inv.logit(alpha_aux_mu_recovered)
        exp(beta_aux_mu_recovered)
        

plot(alpha_aux_mu,(alpha_aux_mu_recovered))
plot(beta_aux_mu,(beta_aux_mu_recovered))
plot(alpha_sigma,inv.logit(alpha_sigma_recovered))
plot(beta_sigma,(beta_sigma_recovered))
plot(tau_alpha_beta,(tau_alpha_beta_recovered))


#individual level parameters (subjects parameters)
alpha_recovered=summary(rl_fit , pars=c("alpha"))$summary[,1] 
beta_recovered=summary(rl_fit , pars=c("beta"))$summary[,1]
plot(true.parms[,1],(alpha_recovered))
plot(true.parms[,2],(beta_recovered))
cor(true.parms[,1],(alpha_recovered))
cor(true.parms[,2],(beta_recovered))

#compare hyperparameters from sample
c(var(logit(alpha_recovered)),var(logit(true.parms[,1])))
c(var(log(beta_recovered)),var(log(true.parms[,2])))

cor(true.parms[,1],true.parms[,2])
cor(alpha_recovered,beta_recovered)
var(alpha_recovered)
var(beta_recovered)


##2nd version
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

#print table to file
df.tbl %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))        







# Arch --------------------------------------------------------------------

tau <- rcauchy(1,0,1);
L_Omega <- chol(2);
Sigma <- diag(tau,L_Omega);


str(df) #checking the data frame 
start_time <- Sys.time()
sim_param<-
  lapply(1:length(Nalt), function(alt) {
    lapply(1:length(Ntrl), function(trl) {
      print(paste('alt=',alt,'trl=',trl))
      
#inserting missing value 
alt=trl=1
df<-  lapply(1:length(Nalt), function(alt) {
  lapply(1:length(Ntrl), function(trl) {
    mclapply(1:Nsubj,function(s)           {
      df_subj=sim.block(Ntrl[trl],Nalt[alt],true.parms$alpha[s],true.parms$beta[s],rndwlk)
      
      #insert missing data
      index  =sample(1:Ntrl[trl],as.integer(runif(1,min=0,max=0.1*Ntrl[trl])))
      df_subj[index,]=NA
    },
    mc.cores=4)})})




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



