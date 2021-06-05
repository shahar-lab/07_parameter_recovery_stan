# Parameter recovery Hierarchical fit Stan

rm(list=ls())

library('rstan') # observe startup messages
library("truncnorm")
library(parallel)
library(gtools) #inv.logit function 

source('sim_Narmed_bandit.R')
rndwlk<-read.csv('rndwlk_4frc_1000trials.csv',header=F)


# simulate data -----------------------------------------------------------
#generate parameters and data for N agents. 
Nsubj   <-50          #number of agents
Nalt<-2               #number of alternatives
Ntrl<-200           #number of trials
pmax.missing_value<-0.1       #max number of simulated missing value 

alpha_mu = 0.5 
alpha_sigma = 0.2 
beta_mu = 3
beta_sigma = 2

true.parms<-data.frame(alpha=rtruncnorm(Nsubj, mean =alpha_mu, sd = alpha_sigma, a = 0,b = 1),beta =rtruncnorm(Nsubj, mean =beta_mu, sd = beta_sigma, a = 0.01,b = 10))
hist(true.parms$alpha)
hist(true.parms$beta)

df<- lapply(1:Nsubj,function(s)           {
                      
                      df_subj=cbind(subj=rep(s,Ntrl),
                                    trial=(1:Ntrl),
                                    sim.block(Ntrl,Nalt,true.parms$alpha[s],true.parms$beta[s],rndwlk))
                      
                      #insert missing data
                      index  =sample(1:Ntrl,as.integer(runif(1,min=0,max=pmax.missing_value*Ntrl)))
                      df_subj[index,-c(1,2)]=NA
                      df_subj
                      })

df<-do.call(rbind,df)


sum(is.na(df))*1 #what is the total number of NAN trails in the data frame


# parameter recovery with stan --------------------------------------------

#prepare action and reward matrices (subject x trial)
a1=t(sapply(1:Nsubj,function(subj) {df[df$subj==subj,'action']}))
reward=t(sapply(1:Nsubj,function(subj) {df[df$subj==subj,'reward']}))

final_trl =list()
final_trl = t(sapply(1:Nsubj,function(subj) {final_trl[subj]=Ntrl-(sum(is.na(a1[subj,]))*1)})) #number of trials without missing values  

source('arrange_MD_forStan.R')
a1=arrange_MD_forStan(mydata=a1,newval=0)
reward=arrange_MD_forStan(mydata=reward,newval=0)      


#fit stan model
      model_data <- list(Nsubj = Nsubj,
                         Ntrials = Ntrl,
                         Narms = Nalt,
                         a1 = a1,
                         reward = reward,
                         final_trl= final_trl)
        
        #options(mc.cores = parallel::detectCores())
        #rstan_options(auto_write=TRUE)
        
        rl_fit<- stan(file = "rl_Hierarchical_mv.stan", data=model_data, iter=2000,chains=4,cores =4) #iter - number of MCMC samples 

        #m = summary(rl_fit , pars=c("alpha","beta"))
        #c(m$summary[1,1],m$summary[2,1])

#population level (hyperparameter)
alpha_population_recovered=summary(rl_fit , pars=c("mu_alpha"))$summary[,1]
beta_population_recovered=summary(rl_fit , pars=c("mu_beta"))$summary[,1]
alpha_tau_population_recovered=summary(rl_fit , pars=c("tau_alpha"))$summary[,1]
beta_tau_population_recovered=summary(rl_fit , pars=c("tau_beta"))$summary[,1]

plot(alpha_mu,inv.logit(alpha_population_recovered))
plot(alpha_sigma,inv.logit(alpha_tau_population_recovered))
plot(beta_mu,exp(beta_population_recovered))
plot(beta_sigma,exp(beta_tau_population_recovered))

#individual level parameters (subjects parameters)
alpha_recovered=summary(rl_fit , pars=c("alpha_subjects"))$summary[,1] #אולי זה צריך להיות alpha[1]
beta_recovered=summary(rl_fit , pars=c("beta_subjects"))$summary[,1]
plot(true.parms[,1],inv.logit(alpha_recovered))
plot(true.parms[,2],exp(beta_recovered))

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



