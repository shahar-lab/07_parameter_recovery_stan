###Parameter recovery individual fit stan

rm(list=ls())

library("rstan") 
library("tidyverse")
library(parallel)
library(gtools) #inv.logit function 

detectCores()


# simulate data -----------------------------------------------------------
source('sim_Narmed_bandit.R')
rndwlk<-read.csv('rndwlk_4frc_1000trials.csv',header=F)

#generate parameters and data for N agents. 
N   =10          #number of agents
Nalt=2           #number of alternatives
Ntrl=200          #number of trials


true.parms<-data.frame(alpha=runif(N, min = 0, max = 1),beta =runif(N, min = 1, max = 8))
df<-  mclapply(1:N,function(s)               {
                   sim.block(Ntrl,Nalt,true.parms$alpha[s],true.parms$beta[s],rndwlk)},
                   mc.cores=4)


# parameter recovery with stan --------------------------------------------

sim_param<-mclapply(1:N,function(subject)   {
      
          
          model_data <- list(nTrials = Ntrl,
                             nArms = Nalt,
                             a1 = df[[subject]]$action,
                             reward = df[[subject]]$reward)

         rl_fit<- stan(file = "rl_basic.stan", data=model_data, iter=1000,chains=4,cores =1) #iter - number of MCMC samples 
         m = summary(rl_fit , pars=c("alpha","beta"))
         c(m$summary[1,1],m$summary[2,1])

  },mc.cores=4)


library("shinystan")
launch_shinystan(rl_fit)







#####ARCH
sim_param_2<-
  lapply(1:length(Nalt), function(alt) {
    lapply(1:length(Ntrl), function(trl) {print(paste('alt=',alt,'trl=',trl))
      mclapply(1:N,function(subject)         {
        
        
        model_data <- list(nTrials = Ntrl[trl],
                           nArms = Nalt[alt],
                           a1 = df[[alt]][[trl]][[subject]]$action,
                           reward = df[[alt]][[trl]][[subject]]$reward)
        
        options(mc.cores = parallel::detectCores())
        rstan_options(auto_write=TRUE)
        
        my_model<- stan_model(file = "rl_basic.stan") 
        sample <- sampling(object = my_model, data = model_data)
        
        #fit <-optimizing(object = my_model, data = model_data)
        #c(fit$par[1],fit$par[2])
        
      },mc.cores=2)
    })
  })


model_data <- list(nTrials = Ntrl[trl],
                   nArms = Nalt[alt],
                   a1 = df[[alt]][[trl]][[subject]]$action,
                   reward = df[[alt]][[trl]][[subject]]$reward)

my_model<- stan_model(file = "rl_basic.stan") 
sample <- sampling(object = my_model, data = model_data)

plot(sample, plotfun = "hist", pars= "alpha")
plot(sample, plotfun = "hist", pars= "beta")

library("shinystan")
launch_shinystan(sample)

#calculate cor between true and recovered params
library(gtools) 
df.tbl   <-lapply(1:length(Nalt), function(alt) {
           lapply(1:length(Ntrl), function(trl) {
                        data.frame(Nalt=Nalt[alt],
                        Ntrl=Ntrl[trl],
                        cor.alpha=cor(true.parms$alpha,inv.logit((do.call(rbind,(sim_param[[alt]][[trl]]))[,1]))),
                        cor.beta=cor(true.parms$beta,exp((do.call(rbind,(sim_param[[alt]][[trl]]))[,2]))))
  })})

df.tbl<-do.call(rbind,lapply(1:length(Nalt), function(alt) {do.call(rbind,df.tbl[[alt]])}))
save(df.tbl, file='rl_individual_fit_0605_48')
#load(file='rl_individual_fit_0405_48_v2')

sim_df=data.frame()

for (alt in length(sim_param)){
  for (trl in length(sim_param[[1]])){
    for (subj in length(sim_param[[1]][[1]])){
    sim_df$alpha <- sim_param[[alt]][[trl]][[subj]][[1]]
    sim_df$beta = sim_param[[alt]][[trl]][[subj]][[2]]
        }
    }
}
  
     
plot(true.parms$alpha,((do.call(rbind,(sim_param[[alt]][[trl]]))[,1])),col='blue')
par(new=TRUE)
plot(true.parms$alpha,inv.logit((do.call(rbind,(sim_param[[alt]][[trl]]))[,1])))

cor.alpha_1=cor(true.parms$alpha,inv.logit((do.call(rbind,(sim_param[[alt]][[trl]]))[,1])))
cor.alpha_2=cor(true.parms$alpha,((do.call(rbind,(sim_param[[alt]][[trl]]))[,1])))

plot(true.parms$beta,((do.call(rbind,(sim_param[[alt]][[trl]]))[,2])),col='blue')
par(new=TRUE)
plot(true.parms$beta,exp((do.call(rbind,(sim_param[[alt]][[trl]]))[,2])),ylim=c(0,8),xlim=c(0,8))
plot(true.parms$beta,exp((do.call(rbind,(sim_param[[alt]][[1]]))[,2])),xlim=c(0,8))
par(new=TRUE)
plot(true.parms$beta,exp((do.call(rbind,(sim_param[[alt]][[2]]))[,2])),xlim=c(0,8),col='blue')

df_compare<- data.frame(
  t <- true.parms$beta,
  s<-exp((do.call(rbind,(sim_param[[alt]][[trl]]))[,2]))
)

#print table to file
df.tbl %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))        


###plotting 
#stan_plot

library("shinystan")
launch_shinystan(sample)
#launch_shinystan_demo()



# Arch --------------------------------------------------------------------
#pre-allocation 
alpha = beta = vector(mode = "list", length =length(Nalt))
for (i in 1:length(alpha)){
  alpha[[i]] = vector(mode = "list", length =length(Ntrl))
  beta[[i]] = vector(mode = "list", length =length(Ntrl))
  for (j in 1:length(alpha[[i]])){
    alpha[[i]][[j]] = vector(mode = "list", length =N)
    beta[[i]][[j]] = vector(mode = "list", length =N)
  }
}

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



