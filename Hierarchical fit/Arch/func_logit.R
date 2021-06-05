mylogit<-function(x){
  x=1/(1+exp(-x))
  return(x)}

mylogit_inv<-function(x){
  x<-log(x/(1-x))
  return(x)}

