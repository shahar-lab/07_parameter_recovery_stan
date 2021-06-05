#this functions recives a dataframe with missing values (NAs) and  
#returns a the data frame where the NAs are at the end of each row
arrange_MD_forStan <-function(mydata,newval){
  MV_trails = list()
  for (subj in seq(1:nrow(mydata))){
    MV_trails[subj]=(sum(is.na(mydata[subj,]))*1)
  }
  df_clean=as.data.frame(matrix(0, ncol = ncol(mydata), nrow = nrow(mydata)))
  for (subj in seq(1:nrow(mydata))){
    df_clean[subj,c((ncol(mydata)-MV_trails[subj][[1]]):ncol(mydata))] <- newval
    df_clean[subj,c(1:(ncol(mydata)-MV_trails[subj][[1]]))] = mydata[subj,][(is.na(mydata[subj,]))==FALSE]
  }
  return(df_clean)
}

