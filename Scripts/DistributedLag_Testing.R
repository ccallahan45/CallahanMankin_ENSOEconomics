# Testing bias in distributed lag models
# in the presence of trends and/or autocorrelation
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

rm(list=ls())
library(ggplot2)
library(margins)
library(sandwich)
library(stargazer)
library(lfe)
library(dplyr)
library(lemon)
library(lspline)
library(fixest)
library(cowplot)
library(gridExtra)
library(tseries)

setwd("/Volumes/rc/lab/C/CMIG/ccallahan/Variability_Economics/Replication/Scripts/") #Data/")
loc_panel <- "../Data/Panel/"
loc_save <- "../Data/DL_Test/"

# helper functions
source("./HelperFunctions.R")

# read in data
y1 <- 1960
y2 <- 2019
panel <- read.csv(paste(loc_panel,"ENSO_Growth_Panel_",y1,"-",y2,".csv",sep=""))
panel$t2 <- (panel$t)**2

# trends
trend_lin <- ""
n_iso <- length(unique(panel$iso))
for (nc in c(0:(n_iso-1))){
  trend_lin <- paste(trend_lin," + yi_linear_",nc,sep="")
}

# add lags and leads
vars_to_lag <- c("e","c","gr_pwt_frac","t","t2")
vars_to_lead <- c("e","c","gr_pwt_frac","t","t2")
for (v in c(1:length(vars_to_lag))){
  var <- vars_to_lag[v]
  print(var)
  for (l in c(1:20)){
    panel %>% group_by(iso) %>% 
      mutate(!!paste(var,"_lag",l,sep="") := dplyr::lag((!!as.name(var)),l)) -> panel
  }
}

for (v in c(1:length(vars_to_lead))){
  var <- vars_to_lead[v]
  print(var)
  for (l in c(1:5)){
    panel %>% group_by(iso) %>% 
      mutate(!!paste(var,"_lead",l,sep="") := dplyr::lead((!!as.name(var)),l)) -> panel
  }
}


cumulative_dl_coef <- function(model,coefs_list){
  coef_ests <- as.vector(coef(summary(model))[coefs_list,"Estimate"])
  cumulative_coef <- sum(coef_ests)
  cumulative_se <- sqrt(sum(vcov(model)[coefs_list,coefs_list]))
  t_stat <- cumulative_coef/cumulative_se
  p <- round(2*pt(-abs(t_stat),df=nobs(model)-1),4)
  ci2_5 <- cumulative_coef - 1.96*cumulative_se
  ci97_5 <- cumulative_coef + 1.96*cumulative_se
  
  return(c(cumulative_coef,cumulative_se,t_stat,p,ci2_5,ci97_5))
}

add_trend <- function(x,trnd){
  # x = timeseries
  # trnd = per-time change in x
  if (sum(!is.na(x))==0){
    x_trend <- x
  } else {
    # start time at zero so trend contributes nothing
    # to first observation of time series
    time <- c(0:(length(x)-1))
    trend <- time*trnd
    x_trend <- x + trend
  }
  return(x_trend)
}

sim_arima <- function(x,ar,sd){
  if (sum(!is.na(x))==0){
    x_arima <- x
  } else {
   x_arima <- arima.sim(n=length(x),list(ar=ar),sd=sd)
  }
  #arima.sim(n=50,list(ar=0.3),sd=0.05)
}

trunc_norm <- function(n,loc,sd,min,max){
  # normal dist truncated at min/max
  # just set the outside values at the mean
  norm <- rnorm(n,loc,sd)
  norm[norm>max] <- rep(loc,length(norm[norm>max]))
  norm[norm<min] <- rep(loc,length(norm[norm<min]))
  return(norm)
}


##### Now do sets of simulations

## we want underlying growth data with noise, trends, 
## autocorrelation, and an ENSO effect

## as well as an increasing number of lags

panel -> sim_data

# effect is negative and then plateaus after 5

# e-index
main_coef_e <- c(0.007,0.002,0.005,0.007,0.004,0.005,0,0,0,
               0,0,0,0,0,0,0,0,0,0,0,0) # similar to actual effect
interaction_coef_e <-  c(-0.015,-0.006,-0.01,-0.01,-0.009,-0.01,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0)
main_coef_sum_e <- sum(main_coef_e)*100
interaction_coef_sum_e <- sum(interaction_coef_e)*100

# now c-index -- one-fourth of e-index effect
main_coef_c <- main_coef_e/4.0
interaction_coef_c <- interaction_coef_e/4.0

# parameters
gr_rand_loc <- 0.02
gr_rand_sd <- 0.05
gr_ar_loc <- 0.1
gr_ar_sd <- 0.2
gr_trend_loc <- 0 
gr_trend_sd <- 0.002
#ar_min <- 0
#ar_max <- 0.95
autocorr_lags <- 0

n_mc <- 1000

for (nlag in c(5:20)){ # start at 5, when the effect plateaus
  coefs_df <- data.frame("boot"=c(1:n_mc),
                         "true_main_coef_e"=numeric(n_mc),
                         "true_int_coef_e"=numeric(n_mc),
                         "main_coef_e"=numeric(n_mc),
                         "int_coef_e"=numeric(n_mc),
                         "true_main_coef_c"=numeric(n_mc),
                         "true_int_coef_c"=numeric(n_mc),
                         "main_coef_c"=numeric(n_mc),
                         "int_coef_c"=numeric(n_mc))
  print(nlag)
  
  for (n in c(1:n_mc)){
    print(n)
    
    sim_data %>% group_by(iso) %>%
      mutate(gr_rand = sim_arima(gr_pwt_frac,gr_ar_loc,gr_rand_sd)) -> 
      sim_data
    
    sim_data %>% group_by(iso) %>%
      mutate(gr_trend = add_trend(gr_rand,rnorm(1,gr_trend_loc,gr_trend_sd))) ->
      sim_data
    
    # loop through lags and calculate change in growth from ENSO
    # assuming a certain effect size 
    for (j in c(1:(nlag+1))){
      #print(j)
      if (j==1){
        gr_effect <- sim_data$e*(main_coef_e[j] + sim_data$t_p_corr_running_e*interaction_coef_e[j]) + 
          sim_data$c*(main_coef_c[j] + sim_data$t_p_corr_running_c*interaction_coef_c[j])
      } else {
        gr_effect <- gr_effect + 
          as.numeric(unlist(sim_data[paste0("e_lag",j-1)]))*(main_coef_e[j] + sim_data$t_p_corr_running_e*interaction_coef_e[j]) + 
          as.numeric(unlist(sim_data[paste0("c_lag",j-1)]))*(main_coef_c[j] + sim_data$t_p_corr_running_c*interaction_coef_c[j])
      }
    }
    
    sim_data$gr_final <- sim_data$gr_trend + gr_effect
    
    var <- "gr_final"
    for (l in c(1:nlag)){
      sim_data %>% group_by(iso) %>% 
        mutate(!!paste(var,"_lag",l,sep="") := dplyr::lag((!!as.name(var)),l)) -> sim_data
    }
    
    # build formula and estimate model
    #enso_var <- "e"
    response_var <- "gr_final"
    fe <- "iso"
    cl <- "0"
    trends <- "" #trend_lin
    dl_coefs_e <- "e"
    for (i in c(1:nlag)){dl_coefs_e <- c(dl_coefs_e,paste0("e_lag",i))}
    dl_coefs_c <- "c"
    for (i in c(1:nlag)){dl_coefs_c <- c(dl_coefs_c,paste0("c_lag",i))}
    
    tc <- "t_p_corr_running"
    dl_coefs_e2 <- paste0("e:",tc,"_e")
    for (i in c(1:nlag)){dl_coefs_e2 <- c(dl_coefs_e2,paste0(tc,"_e:e_lag",i))}
    dl_coefs_c2 <- paste0("c:",tc,"_c")
    for (i in c(1:nlag)){dl_coefs_c2 <- c(dl_coefs_c2,paste0(tc,"_c:c_lag",i))}
    
    form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
    for (j in c(1:nlag)){
      form_dl <- paste0(form_dl," + e_lag",j," + e_lag",j,"*",tc,"_e",
                        " + c_lag",j," + c_lag",j,"*",tc,"_c")
    }
    if (autocorr_lags>0){
      for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
    }
    
    form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
    mdl <- felm(form,data=sim_data)
    coefs_df[n,"main_coef_e"] <- sum(coef(mdl)[dl_coefs_e])
    coefs_df[n,"int_coef_e"] <- sum(coef(mdl)[dl_coefs_e2])
    coefs_df[n,"main_coef_c"] <- sum(coef(mdl)[dl_coefs_c])
    coefs_df[n,"int_coef_c"] <- sum(coef(mdl)[dl_coefs_c2])
    
    coefs_df[n,"true_main_coef_e"] <- sum(main_coef_e)
    coefs_df[n,"true_int_coef_e"] <- sum(interaction_coef_e)
    coefs_df[n,"true_main_coef_c"] <- sum(main_coef_c)
    coefs_df[n,"true_int_coef_c"] <- sum(interaction_coef_c)
  }
  
  fname <- paste0("distributed_lag_test_laglength_",nlag,"_science_r1.csv")
  write.csv(coefs_df,paste(loc_save,fname,sep=""))
  print(fname)
}

