##simulate data frame with NANs 
rm(list=ls())

Nsubj = 10 #number of subjects 
Ntrails = 500 #number of trails per subject 
Nnans = 0.1 #total amount of NAN trails from the hall data frame  

#creating the data frame
df <- as.data.frame(matrix(1, ncol = Ntrails, nrow = Nsubj))

##inserting the Nans 
#tot=Nnans*Ntrails
for (subj in seq(1:Nsubj)){
  print(subj)
  df[subj,][sample(ncol(df),as.integer(runif(1,min=0,max=Nnans*Ntrails)))] <- NA
  #x= runif(1, min=0, max=tot-x)
}

sum(is.na(df))*1 #what is the total number of NAN trails in the data frame

#creating a list of the number of NAN trails of each subject
Nnan_trails = list()
for (subj in seq(1:Nsubj)){
  Nnan_trails[subj]=(sum(is.na(df[subj,]))*1)
}

#creating a new data frame where the NANs trails of each subjects are in the end of the row

df_clean=as.data.frame(matrix(0, ncol = Ntrails, nrow = Nsubj))
for (subj in seq(1:Nsubj)){
  print(subj)
  df_clean[subj,c((Ntrails-Nnan_trails[subj][[1]]):Ntrails)] <- NA
  df_clean[subj,c(1:(Ntrails-Nnan_trails[subj][[1]]))] = df[subj,][(is.na(df[subj,]))==FALSE]
}



# ARCH --------------------------------------------------------------------

df$V1[sample(nrow(df),3)] <- NA
df[,2][sample(nrow(df),3)] <- NA
df[2,][sample(ncol(df),500)] <- NA
df[4,][sample(ncol(df),seq(0:Ntrails*0.1))] <- NA
sum(is.na(df[4,]))*1

set.seed(211)
df[-1] <- sapply(df[-1], function(x) ifelse(runif(length(x)) < Nnans, NA , x))
sum(is.na(df))*1

df[1,] <- sapply(df[1,], function(x) ifelse(runif(length(x)) < Nnans, NA , x))
sum(is.na(df[1,]))*1


df[sample(as.integer(df[1,]),3)] <- NA
df[sample(nrow(df[1,]),3)] <- NA

