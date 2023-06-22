# Relationship between ENSO and economic growth
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
library(texreg)

# location
# setwd("/your/desired/path/")
loc_panel <- "../Data/Panel/"
loc_save <- "../Data/RegressionResults/"

# helper functions
source("./HelperFunctions.R")

# read in data
y1 <- 1960
y2 <- 2019
panel <- read.csv(paste(loc_panel,"ENSO_Growth_Panel_",y1,"-",y2,".csv",sep=""))

# add vars
panel$t2 <- panel$t**2
panel$p2 <- panel$p**2

# other
panel$growth_wb <- panel$growth
panel$etau <- panel$e * panel$t_p_corr_running_e
panel$ctau <- panel$c * panel$t_p_corr_running_c

# trends
trend_lin <- ""
trend_quad <-  ""
n_iso <- length(unique(panel$iso))
for (nc in c(0:(n_iso-1))){
  trend_lin <- paste(trend_lin," + yi_linear_",nc,sep="")
  trend_quad <- paste(trend_quad," + yi_quadratic_",nc,sep="")
}
  
# add lags and leads
vars_to_lag <- c("e","c","p","p2","t","t2",
                   "gr_pwt_frac","gr_cs_frac","gr_tfp_frac",
                   "growth_wb","nino3","etau","ctau","nino34")
vars_to_lead <- c("e","c","t","t2","etau","ctau")
  
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


##################
#### How many outliers are we removing?
##################

min_gr <- -100 # outlier limitation
max_gr <- 100 # 0.18 = approx 3 sigma
panel %>% 
  filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,
         !is.na(e),!is.na(t_p_corr_running_e)) -> panel1
print(dim(panel1)[1])
min_gr <- -0.18 # outlier limitation
max_gr <- 0.18 # 0.18 = approx 3 sigma
panel %>% 
  filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,
         !is.na(e),!is.na(t_p_corr_running_e)) -> panel2
print(dim(panel2)[1])

dimdiff <- dim(panel1)[1] - dim(panel2)[1]
print(dimdiff)
print(100*dimdiff/(dim(panel1)[1]))


##################
#### Plots/initial tables
##################


# some quick correlations
cor_data <- panel %>% filter(iso=="USA",year>=1965) %>% 
  ungroup() %>%
  select(e,e_lag1,e_lag2,e_lag3,e_lag4,e_lag5)
cor(cor_data,method="pearson")
cor(panel[panel$iso=="USA",]$e,panel[panel$iso=="USA",]$c)
summary(lm("c ~ e",panel[panel$iso=="USA",]))
ar_mdl <- lm("e ~ e_lag1 + e_lag2 + e_lag3 + e_lag4 + e_lag5",panel[panel$iso=="USA",])
summary(ar_mdl)
texreg(list(ar_mdl),digits=4,stars=c(0.001,0.01,0.05))


cor_data2 <- panel %>% filter(iso=="USA",year>=1965) %>% 
  ungroup() %>%
  select(e,e_lag1,e_lag2,e_lag3,e_lag4,e_lag5,
         e_lag6,e_lag7,e_lag8,e_lag9,e_lag10,
         e_lag11,e_lag12,e_lag13,e_lag14,e_lag15,
         e_lag16,e_lag17,e_lag18,e_lag19,e_lag20)
cor(cor_data2,method="pearson",use="complete.obs")


##################
#### Regression functions
##################

cumulative_dl_coef <- function(model,coefs_list){
  coef_ests <- as.vector(coef(summary(model))[coefs_list,"Estimate"])
  cumulative_coef <- sum(coef_ests)
  cumulative_se <- sqrt(sum(vcov(model)[coefs_list,coefs_list]))  
  t_stat <- cumulative_coef/cumulative_se
  p <- round(2*pt(-abs(t_stat),df=nobs(model)-1),4)
  ci2_5 <- cumulative_coef - 1.96*cumulative_se
  ci97_5 <- cumulative_coef + 1.96*cumulative_se
  print(paste("coef = ",round(cumulative_coef,4),
              ", se = ",round(cumulative_se,4),
              ", p = ",round(p,4),sep=""))
  return(c(cumulative_coef,cumulative_se,t_stat,p,ci2_5,ci97_5))
}

####





##################
#### Initial set of regressions 
##################


n_config <- 6
configurations <- data.frame("enso_var"=numeric(n_config),
                             "response_var"=numeric(n_config),
                             "trends"=numeric(n_config),
                             "teleconnection"=numeric(n_config))
configurations[1,] <- c("e_and_c","gr_pwt_frac","none","t_p_corr_running")
configurations[2,] <- c("nino3","gr_pwt_frac","none","t_p_corr_running")
configurations[3,] <- c("e_and_c","growth_wb","none","t_p_corr_running")
configurations[4,] <- c("e_and_c","gr_deviation","none","t_p_corr_running")
configurations[5,] <- c("e","gr_pwt_frac","none","t_p_corr_running")
configurations[6,] <- c("nino34","gr_pwt_frac","none","t_p_corr_running")

## Heterogeneous effects by teleconnection strength
nlag <- 5
fe <- "iso"
cl <- "0"
nboot <- 1000
autocorr_lags <- 0
min_gr <- -0.18 # outlier limitation
max_gr <- 0.18 # 0.18 = approx 3 sigma

for (c in c(1:n_config)){
  set.seed(100) # make sure it's the same bootstraps across variables
  print(paste("configuration: ",c,sep=""))
  enso_var <- configurations[c,"enso_var"]
  if ((enso_var=="nino3")|(enso_var=="nino34")){enso_tc<-"e"}else{enso_tc<-enso_var}
  response_var <- configurations[c,"response_var"]
  trnds <- configurations[c,"trends"]
  tc <- configurations[c,"teleconnection"]
  if (trnds=="none"){trends=""}else if(trnds=="linear"){trends=trend_lin}else{trends=""}
  
  # filter outliers and calculate growth deviation  
  panel %>% 
    filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) %>%
    group_by(year) %>% 
    mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
    mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
  
  # set up dataframe
  enso_coefs_dist <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  if (enso_var=="e_and_c"){
    # secondary set of ENSO effects to get C index as well
    enso_coefs_dist2 <- data.frame("boot"=c(1:nboot))
    exp_interact_coefs_dist2 <- data.frame("boot"=c(1:nboot))
    for (ll in c(0:nlag)){
      enso_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
      exp_interact_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
    }
  }
  
  # loop
  for (n in c(1:nboot)){
    print(n)
    # bootstrap by country
    
    isos <- as.vector(unique(dat$iso))
    iso_boot <- sample(isos,size=length(isos),replace=T)
    df_boot <- sapply(iso_boot, function(x) which(dat[,'iso']==x))
    data_boot <- dat[unlist(df_boot),]

    
    # run model for each set of lags
    if (enso_var=="e_and_c"){
      form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
      for (j in c(1:nlag)){
        form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                          " + c_lag",j," + ","c_lag",j,"*",tc,"_c")
      }
    } else {
      form_dl <- paste0(response_var," ~ ",enso_var," + ",enso_var,"*",tc,"_",enso_tc)
      for (j in c(1:nlag)){
        form_dl <- paste0(form_dl," + ",enso_var,"_lag",j," + ",
                          enso_var,"_lag",j,"*",tc,"_",enso_tc)
      }
    }
    
    if (autocorr_lags>0){
      for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
    }
    
    form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
    mdl <- felm(form,data=data_boot)
    
    # extract coefficients
    if (enso_var=="e_and_c"){
      enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["e"])
      exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("e:",tc,"_e")])
      for (ll in c(1:nlag)){
        enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("e_lag",ll)])
        exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_e:e_lag",ll)])
      }
      
      enso_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)["c"])
      exp_interact_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("c:",tc,"_c")])
      for (ll in c(1:nlag)){
        enso_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("c_lag",ll)])
        exp_interact_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_c:c_lag",ll)])
      }
    } else {
      enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[enso_var])
      exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0(enso_var,":",tc,"_",enso_tc)])
      for (ll in c(1:nlag)){
        enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(enso_var,"_lag",ll)])
        exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_",enso_tc,":",enso_var,"_lag",ll)])
      }
    }
  }
  
  
  
  if (enso_var=="e_and_c"){
    # write out ENSO coefs
    fname_enso_dist1 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                              response_var,"_trend",trnds,".csv")
    write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist1,sep=""))
    print(fname_enso_dist1)
    fname_enso_dist2 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                               response_var,"_trend",trnds,".csv")
    write.csv(enso_coefs_dist2,paste(loc_save,fname_enso_dist2,sep=""))
    print(fname_enso_dist2)
    
    # write out interaction coefs
    fname_int_dist1 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                             response_var,"_trend",trnds,".csv")
    write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist1,sep=""))
    print(fname_int_dist1)
    fname_int_dist2 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                             response_var,"_trend",trnds,".csv")
    write.csv(exp_interact_coefs_dist2,paste(loc_save,fname_int_dist2,sep=""))
    print(fname_int_dist2)
    
    
  } else {
    # write out ENSO coefs
    fname_enso_dist <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_",enso_var,"_",
                              response_var,"_trend",trnds,".csv")
    write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist,sep=""))
    print(fname_enso_dist)
    
    # write out interaction coefs
    fname_int_dist <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_",enso_var,"_",
                             response_var,"_trend",trnds,".csv")
    write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist,sep=""))
    print(fname_int_dist)
  }
}

####




##################
#### Longer lag times
##################

nlags <- c(10:20) #c(10:20)
fe <- "iso" #+ year
cl <- "0"
nboot <- 1000
min_gr <- -0.18 # outlier limitation
max_gr <- 0.18 # 0.18 = approx 3 sigma
autocorr_lags <- 0

for (nlag in nlags){ 
  print(nlag)
  set.seed(100)
  for (c in c(1)){ 
    set.seed(100) # make sure it's the same bootstraps in different configurations
    print(paste("configuration: ",c,sep=""))
    enso_var <- configurations[c,"enso_var"]
    if (enso_var=="nino3"){enso_tc<-"e"}else{enso_tc<-enso_var}
    response_var <- configurations[c,"response_var"]
    trnds <- configurations[c,"trends"]
    tc <- configurations[c,"teleconnection"]
    if (trnds=="none"){trends=""}else if(trnds=="linear"){trends=trend_lin}else{trends=""}
    
    # filter outliers?  
    panel %>% 
      filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    
    # set up dataframe
    enso_coefs_dist <- data.frame("boot"=c(1:nboot))
    exp_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
    for (ll in c(0:nlag)){
      enso_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
      exp_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    }
    
    if (enso_var=="e_and_c"){
      # secondary set of ENSO effects to get C index as well
      enso_coefs_dist2 <- data.frame("boot"=c(1:nboot))
      exp_interact_coefs_dist2 <- data.frame("boot"=c(1:nboot))
      for (ll in c(0:nlag)){
        enso_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
        exp_interact_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
      }
    }
    
    # loop
    for (n in c(1:nboot)){
      print(n)
      # bootstrap by country
      
      isos <- as.vector(unique(dat$iso))
      iso_boot <- sample(isos,size=length(isos),replace=T)
      df_boot <- sapply(iso_boot, function(x) which(dat[,'iso']==x))
      data_boot <- dat[unlist(df_boot),]
      
      # run model for each set of lags
      if (enso_var=="e_and_c"){
        form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
        for (j in c(1:nlag)){
          form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                            " + c_lag",j," + ","c_lag",j,"*",tc,"_c")
        }
      } else {
        form_dl <- paste0(response_var," ~ ",enso_var," + ",enso_var,"*",tc,"_",enso_tc)
        for (j in c(1:nlag)){
          form_dl <- paste0(form_dl," + ",enso_var,"_lag",j," + ",
                            enso_var,"_lag",j,"*",tc,"_",enso_tc)
        }
      }
      
      if (autocorr_lags>0){
        for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
      }
      
      form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
      mdl <- felm(form,data=data_boot)
      
      # extract coefficients
      if (enso_var=="e_and_c"){
        enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["e"])
        exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("e:",tc,"_e")])
        for (ll in c(1:nlag)){
          enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("e_lag",ll)])
          exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_e:e_lag",ll)])
        }
        
        enso_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)["c"])
        exp_interact_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("c:",tc,"_c")])
        for (ll in c(1:nlag)){
          enso_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("c_lag",ll)])
          exp_interact_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_c:c_lag",ll)])
        }
      } else {
        enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[enso_var])
        exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0(enso_var,":",tc,"_",enso_tc)])
        for (ll in c(1:nlag)){
          enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(enso_var,"_lag",ll)])
          exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_",enso_tc,":",enso_var,"_lag",ll)])
        }
      }
      
    }
    
    if (enso_var=="e_and_c"){
      # write out ENSO coefs
      fname_enso_dist1 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                                 response_var,"_trend",trnds,".csv")
      write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist1,sep=""))
      print(fname_enso_dist1)
      fname_enso_dist2 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                                 response_var,"_trend",trnds,".csv")
      write.csv(enso_coefs_dist2,paste(loc_save,fname_enso_dist2,sep=""))
      print(fname_enso_dist2)
      
      # write out interaction coefs
      fname_int_dist1 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                                response_var,"_trend",trnds,".csv")
      write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist1,sep=""))
      print(fname_int_dist1)
      fname_int_dist2 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                                response_var,"_trend",trnds,".csv")
      write.csv(exp_interact_coefs_dist2,paste(loc_save,fname_int_dist2,sep=""))
      print(fname_int_dist2)
      
      
    } else {
      # write out ENSO coefs
      fname_enso_dist <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_",enso_var,"_",
                                response_var,"_trend",trnds,".csv")
      write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist,sep=""))
      print(fname_enso_dist)
      
      # write out interaction coefs
      fname_int_dist <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_",enso_var,"_",
                               response_var,"_trend",trnds,".csv")
      write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist,sep=""))
      print(fname_int_dist)
    }
  }
}

####








##################  
#### Additional assorted sensitivity tests
##################


nlag <- 5
#fe <- "iso"
cl <- "0"
nboot <- 1000
set.seed(100)
autocorr_lags <- 0

n_sens <- 11
# trends, no extreme, no teleconnections>1, no outlier removal, 
# excluding temp covariate, adding precip control
# splitting data by p_corr2_e > 0  or < 0,
# excluding fixed effect,
# both linear and quadratic trends
# including year FE
min_gr <- -0.18 # outlier limitation
max_gr <- 0.18

for (s in c(1:n_sens)){
  
  if (s==1){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    trends <- trend_lin
    trnds <- "linear"
    sens <- "trends"
    tc <- "t_p_corr_running"
    fe <- "iso"
  } else if (s==2) {
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    #dat %>% filter(e<3,e_lag1<3,e_lag2<3,e_lag3<3,e_lag4<3,e_lag5<3) -> dat
    dat %>% filter(year!=1983,year!=1998) -> dat
    sens <- "noextreme"
    trends <- ""
    trnds <- "none"
    tc <- "t_p_corr_running"
    fe <- "iso"
  } else if (s==3) {
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    dat %>% filter(t_p_corr_running_e<0.8) -> dat
    sens <- "teleconnectionlimited"
    trends <- ""
    trnds <- "none"
    tc <- "t_p_corr_running"
    fe <- "iso"
  } else if (s==4){
    panel %>% filter(gr_pwt_frac<1,gr_pwt_frac>-1) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    trends <- "" 
    trnds <- "none"
    sens <- "outliers"
    tc <- "t_p_corr_running"
    fe <- "iso"
  } else if (s==5){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "tempcontrol"
    tc <- "t_p_corr_running"
    trends <- ""
    trnds <- "none"
    fe <- "iso"
  } else if (s==6){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "precipcontrol"
    tc <- "t_p_corr_running"
    trends <- ""
    trnds <- "none" 
    fe <- "iso"
  } else if (s==7){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,p_corr2_e>0) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "positiveprecip"
    tc <- "t_p_corr_running"
    trends <- ""
    trnds <- "none"
    fe <- "iso"
  } else if (s==8){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,p_corr2_e<0) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "negativeprecip"
    tc <- "t_p_corr_running"
    trends <- ""
    trnds <- "none"
    fe <- "iso"
  } else if (s==9){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "nofe"
    trends <- ""
    trnds <- "none"
    tc <- "t_p_corr_running"
    fe <- "0"
  } else if (s==10){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "trends2"
    tc <- "t_p_corr_running"
    trends <- paste0(trend_lin,trend_quad)
    trnds <- "linquad"
    fe <- "iso"
  } else if (s==11){
    panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
    dat %>% group_by(year) %>% 
      mutate(gr_yrmean = mean(gr_pwt_frac,na.rm=T)) %>%
      mutate(gr_deviation = gr_pwt_frac - gr_yrmean) -> dat
    sens <- "yearfe"
    tc <- "t_p_corr_running"
    trends <- ""
    trnds <- "none"
    fe <- "iso + year"
  }
  
  
  print(paste0("sensitivity: ",sens))
  response_var <- "gr_pwt_frac"
  
  # set up dataframe for e- and c-indices
  enso_coefs_dist <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  enso_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  # loop
  for (n in c(1:nboot)){
    print(n)
    # bootstrap by country
    
    isos <- as.vector(unique(dat$iso))
    iso_boot <- sample(isos,size=length(isos),replace=T)
    df_boot <- sapply(iso_boot, function(x) which(dat[,'iso']==x))
    data_boot <- dat[unlist(df_boot),]
    
    # build and run distributed lag model
    form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
    for (j in c(1:nlag)){
      form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                        " + c_lag",j," + ","c_lag",j,"*",tc,"_c")
    }
    if (sens=="tempcontrol"){
      form_dl <- paste0(form_dl," + t + t2")
      for (j in c(1:nlag)){form_dl <- paste0(form_dl," + t_lag",j," + t2_lag",j)}
    }
    if (sens=="precipcontrol"){
      form_dl <- paste0(form_dl," + p + p2")
      for (j in c(1:nlag)){form_dl <- paste0(form_dl," + p_lag",j," + p2_lag",j)}
    }
    
    if (autocorr_lags>0){
      for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
    }
    form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
    mdl <- felm(form,data=data_boot)
    
    # extract coefficients
    enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["e"])
    exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("e:",tc,"_e")])
    for (ll in c(1:nlag)){
      enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("e_lag",ll)])
      exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_e:e_lag",ll)])
    }
    
    enso_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)["c"])
    exp_interact_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("c:",tc,"_c")])
    for (ll in c(1:nlag)){
      enso_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("c_lag",ll)])
      exp_interact_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_c:c_lag",ll)])
    }
    
  }
  
  # write out ENSO coefs
  fname_enso_dist1 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                             response_var,"_trend",trnds,"_sensitivity_",sens,".csv")
  write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist1,sep=""))
  print(fname_enso_dist1)
  fname_enso_dist2 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                             response_var,"_trend",trnds,"_sensitivity_",sens,".csv")
  write.csv(enso_coefs_dist2,paste(loc_save,fname_enso_dist2,sep=""))
  print(fname_enso_dist2)
  
  # write out interaction coefs
  fname_int_dist1 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                            response_var,"_trend",trnds,"_sensitivity_",sens,".csv")
  write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist1,sep=""))
  print(fname_int_dist1)
  fname_int_dist2 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                            response_var,"_trend",trnds,"_sensitivity_",sens,".csv")
  write.csv(exp_interact_coefs_dist2,paste(loc_save,fname_int_dist2,sep=""))
  print(fname_int_dist2)
  
}

####












##################  
#### Alternative teleconnections
##################

nlag <- 5
fe <- "iso"
cl <- "0"
nboot <- 1000
set.seed(100)
autocorr_lags <- 0

tcs <- c("t_p_reg_running","t_corr_running","p_corr_running",
         "t_p_corr_sum","t_p_corr_sum_sig")
tc_names <- c("regrunning","temponly","preciponly","sum","sigsum")
n_tc <- length(tcs)
min_gr <- -0.18 # outlier limitation
max_gr <- 0.18


for (t in c(1:n_tc)){
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
  
  trends <- ""
  trnds <- "none"
  tc <- tcs[t]
  tcnm <- tc_names[t]
  response_var <- "gr_pwt_frac"
  
  print(paste0("teleconnections: ",tcnm))
  
  # set up dataframe for e- and c-indices
  enso_coefs_dist <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  enso_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  # loop
  for (n in c(1:nboot)){
    print(n)
    # bootstrap by country
    
    isos <- as.vector(unique(dat$iso))
    iso_boot <- sample(isos,size=length(isos),replace=T)
    df_boot <- sapply(iso_boot, function(x) which(dat[,'iso']==x))
    data_boot <- dat[unlist(df_boot),]
    
    # build and run distributed lag model
    form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
    for (j in c(1:nlag)){
      form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                        " + c_lag",j," + ","c_lag",j,"*",tc,"_c")
    }
    
    #form_dl <- paste0(form_dl," + t + t2")
    #for (j in c(1:nlag)){form_dl <- paste0(form_dl," + t_lag",j," + t2_lag",j)}
    
    if (autocorr_lags>0){
      for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
    }
    form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
    mdl <- felm(form,data=data_boot)
    
    # extract coefficients
    enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["e"])
    exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("e:",tc,"_e")])
    for (ll in c(1:nlag)){
      enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("e_lag",ll)])
      exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_e:e_lag",ll)])
    }
    
    enso_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)["c"])
    exp_interact_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("c:",tc,"_c")])
    for (ll in c(1:nlag)){
      enso_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("c_lag",ll)])
      exp_interact_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_c:c_lag",ll)])
    }
    
  }
  
  # write out ENSO coefs
  fname_enso_dist1 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                             response_var,"_trend",trnds,"_teleconnections_",tcnm,".csv")
  write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist1,sep=""))
  print(fname_enso_dist1)
  fname_enso_dist2 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                             response_var,"_trend",trnds,"_teleconnections_",tcnm,".csv")
  write.csv(enso_coefs_dist2,paste(loc_save,fname_enso_dist2,sep=""))
  print(fname_enso_dist2)
  
  # write out interaction coefs
  fname_int_dist1 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                            response_var,"_trend",trnds,"_teleconnections_",tcnm,".csv")
  write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist1,sep=""))
  print(fname_int_dist1)
  fname_int_dist2 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                            response_var,"_trend",trnds,"_teleconnections_",tcnm,".csv")
  write.csv(exp_interact_coefs_dist2,paste(loc_save,fname_int_dist2,sep=""))
  print(fname_int_dist2)
  
}

####









##################  
#### Additional bootstrap/clustering types
##################


nlag <- 5
enso_var <- "e"
response_var <- "gr_pwt_frac"
fe <- "iso"
cl <- "0"
nboot <- 1000
autocorr_lags <- 0
tc <- "t_p_corr_running"
trends <- "" #trend_lin #""
trnds <- "none"
set.seed(100)

# five-year blocks (burke et al 2018)
# also check removing outliers here...given that including them
# increases uncertainty
panel %>% rowwise() %>% 
  mutate(block=round((year+2)/5)*5) %>%
  mutate(region_block=paste0(region,"_",block)) -> panel

boot_types <- c("year","yr_reg","region","block")
# should all be strings that correspond to 
# column names in data

min_gr <- -0.18 # outlier limitation
max_gr <- 0.18

for (b in c(1:length(boot_types))){
  bt <- boot_types[b]
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
  print(paste0("bootstrap: ",bt))
  
  # set up dataframe
  enso_coefs_dist <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  enso_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  # loop
  for (n in c(1:nboot)){
    print(n)
    
    boot_col_array <- as.numeric(unlist(unique(dat[,bt])))
    resample_array <- sample(boot_col_array,size=length(boot_col_array),replace=T)
    df_boot <- sapply(resample_array, function(x) which(dat[,bt]==x))
    data_boot <- dat[unlist(df_boot),]
    
    form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
    for (j in c(1:nlag)){
      form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                        " + c_lag",j," + ","c_lag",j,"*",tc,"_c")
    }

    if (autocorr_lags>0){
      for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
    }
    
    form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
    mdl <- felm(form,data=data_boot)
    
    # extract coefficients
    enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["e"])
    exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("e:",tc,"_e")])
    for (ll in c(1:nlag)){
      enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("e_lag",ll)])
      exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_e:e_lag",ll)])
    }
    
    enso_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)["c"])
    exp_interact_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("c:",tc,"_c")])
    for (ll in c(1:nlag)){
      enso_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("c_lag",ll)])
      exp_interact_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_c:c_lag",ll)])
    }
  }
  
  # write out ENSO coefs
  fname_enso_dist1 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                             response_var,"_trend",trnds,"_bootstrap_",bt,".csv")
  write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist1,sep=""))
  print(fname_enso_dist1)
  fname_enso_dist2 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                             response_var,"_trend",trnds,"_bootstrap_",bt,".csv")
  write.csv(enso_coefs_dist2,paste(loc_save,fname_enso_dist2,sep=""))
  print(fname_enso_dist2)
  
  # write out interaction coefs
  fname_int_dist1 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                            response_var,"_trend",trnds,"_bootstrap_",bt,".csv")
  write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist1,sep=""))
  print(fname_int_dist1)
  fname_int_dist2 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                            response_var,"_trend",trnds,"_bootstrap_",bt,".csv")
  write.csv(exp_interact_coefs_dist2,paste(loc_save,fname_int_dist2,sep=""))
  print(fname_int_dist2)
  
}

####




##################  
#### Heterogeneity by income group
##################

# set up regression
nlag <- 5
enso_var <- "e"
response_var <- "gr_pwt_frac"
fe <- "iso"
cl <- "0"
nboot <- 1000
autocorr_lags <- 0
tc <- "t_p_corr_running" #"t_p_reg"
trends <- "" #""
trnds <- "none"
set.seed(100)
min_gr <- -0.18 # outlier limitation
max_gr <- 0.18

groups <- c("low","high")
for (g in c(1:length(groups))){
  group <- groups[g]
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,
                   income_group==group) -> dat
  
  print(paste0("group: ",group,"-income"))
  
  # set up dataframe for e- and c-indices
  enso_coefs_dist <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  enso_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  exp_interact_coefs_dist2 <- data.frame("boot"=c(1:nboot))
  for (ll in c(0:nlag)){
    enso_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
    exp_interact_coefs_dist2[paste0("coef_lag",ll)] <- numeric(nboot)
  }
  
  # loop
  for (n in c(1:nboot)){
    print(n)
    
    # bootstrap by country
    
    isos <- as.vector(unique(dat$iso))
    iso_boot <- sample(isos,size=length(isos),replace=T)
    df_boot <- sapply(iso_boot, function(x) which(dat[,'iso']==x))
    data_boot <- dat[unlist(df_boot),]
    
    # build and run distributed lag model
    form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
    for (j in c(1:nlag)){
      form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                        " + c_lag",j," + ","c_lag",j,"*",tc,"_c")
    }
    
    #form_dl <- paste0(form_dl," + t + t2")
    #for (j in c(1:nlag)){form_dl <- paste0(form_dl," + t_lag",j," + t2_lag",j)}
    
    if (autocorr_lags>0){
      for (j in c(1:autocorr_lags)){form_dl <- paste0(form_dl," + ",response_var,"_lag",j)}
    }
    form <- as.formula(paste0(form_dl,trends,"| ",fe," | 0 | ",cl))
    mdl <- felm(form,data=data_boot)
    
    # extract coefficients
    enso_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["e"])
    exp_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("e:",tc,"_e")])
    for (ll in c(1:nlag)){
      enso_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("e_lag",ll)])
      exp_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_e:e_lag",ll)])
    }
    
    enso_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)["c"])
    exp_interact_coefs_dist2[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0("c:",tc,"_c")])
    for (ll in c(1:nlag)){
      enso_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("c_lag",ll)])
      exp_interact_coefs_dist2[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(tc,"_c:c_lag",ll)])
    }
    
  }
  
  # write out ENSO coefs
  fname_enso_dist1 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                             response_var,"_trend",trnds,"_",group,"income.csv")
  write.csv(enso_coefs_dist,paste(loc_save,fname_enso_dist1,sep=""))
  print(fname_enso_dist1)
  fname_enso_dist2 <- paste0("ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                             response_var,"_trend",trnds,"_",group,"income.csv")
  write.csv(enso_coefs_dist2,paste(loc_save,fname_enso_dist2,sep=""))
  print(fname_enso_dist2)
  
  # write out interaction coefs
  fname_int_dist1 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_e_e-and-c_",
                            response_var,"_trend",trnds,"_",group,"income.csv")
  write.csv(exp_interact_coefs_dist,paste(loc_save,fname_int_dist1,sep=""))
  print(fname_int_dist1)
  fname_int_dist2 <- paste0("ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag",nlag,"_c_e-and-c_",
                            response_var,"_trend",trnds,"_",group,"income.csv")
  write.csv(exp_interact_coefs_dist2,paste(loc_save,fname_int_dist2,sep=""))
  print(fname_int_dist2)
  
}

####







##################  
#### Info criterion for lag length choice
##################

lag_lengths <- c(1:20)
min_gr <- -0.18
max_gr <- 0.18
fe <- "iso"
cl <- "0"
response_var <- "gr_pwt_frac"
tc <- "t_p_corr_running" 
aic_df <- data.frame(lags=lag_lengths,
                     aic=length(lag_lengths),
                     dof=length(lag_lengths))


for (ll in c(1:length(lag_lengths))){
  nlag <- lag_lengths[ll]
  
  print(nlag)
  
  # filter outliers?
  # and use exact same data for all estimations
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,
                   year>=1971,year<=2019) -> dat
  
  # set up formula
  form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
  for (j in c(1:nlag)){
    form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                      " + c_lag",j," + ","c_lag",j,"*",tc,"_c")  
  }
  
  #form_dl <- paste0(form_dl," + t + t2")
  #for (j in c(1:nlag)){form_dl <- paste0(form_dl," + t_lag",j," + t2_lag",j)}
  
  form <- as.formula(paste0(form_dl," | ",fe," | 0 | ",cl))
  mdl <- felm(form,data=dat)
  if (nlag==5){
    form_cluster <- as.formula(paste0(form_dl," | ",fe," | 0 | yr_reg"))
    mdl_cluster <- felm(form_cluster,data=dat)
    dl_coefs <- c("e","e_lag1","e_lag2","e_lag3","e_lag4","e_lag5")
    dl_coefs_interact <- c("e:",tc,"_e",paste0(tc,"_e:e_lag1"),paste0(tc,"_e:e_lag2"),
                           paste0(tc,"_e:e_lag3"),paste0(tc,"_e:e_lag4"),paste0(tc,"_e:e_lag5"))
    #print(cumulative_dl_coef(mdl,dl_coefs))
    #print(cumulative_dl_coef(mdl,dl_coefs_interact))
    #print(cumulative_dl_coef(mdl_cluster,dl_coefs))
    #print(cumulative_dl_coef(mdl_cluster,dl_coefs_interact))
  }
  
  aic_df[ll,"aic"] <- AIC(mdl)
  aic_df[ll,"dof"] <- mdl$df
  
  #aic_vals[ll] <- print(AIC(mdl))
}


write.csv(aic_df,paste(loc_save,"ENSO_distributed_lag_e-and-c_AIC.csv",sep=""))

####






















##################  
#### Table for parametric standard errors
##################

min_gr <- -0.18
max_gr <- 0.18
fe <- "iso"
response_var <- "gr_pwt_frac"
tc <- "t_p_corr_running"
nlag <- 5
form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
for (j in c(1:nlag)){
  form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                    " + c_lag",j," + ","c_lag",j,"*",tc,"_c")  
}

panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
dat %>% rowwise() %>% 
  mutate(block=round((year+2)/5)*5) %>%
  mutate(region_block=paste0(region,"_",block)) -> dat


## model 1: ckuster by country
cl <- "iso"
mdl1 <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)

## model 2: cluster by year-continent
cl <- "yr_reg"
mdl2 <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)

## model 3: cluster by year
cl <- "year"
mdl3 <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)

## model 4: cluster by continent
cl <- "region"
mdl4 <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)

## model 5: cluster by five-year block
cl <- "block"
mdl5 <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)


## write out table
texreg(list(mdl1,mdl2,mdl3,mdl4,mdl5),digits=4,
       omit.coef="(c_lag)|(t_lag)|(t2_lag)",stars=c(0.001,0.01,0.05))

## same table but for C-index instead
texreg(list(mdl1,mdl2,mdl3,mdl4,mdl5),digits=4,
       omit.coef="(e_lag)|(t_lag)|(t2_lag)",stars=c(0.001,0.01,0.05))

####






##################  
#### Year effects tests
##################


# first, simply discretize tau into "treated" and "non-treated" groups
# then estimate DL regression with time and country FEs
# discretization done with respect to the E-index teleconnections
# since we care more about that than the C-index

min_gr <- -0.18
max_gr <- 0.18
panel %>% 
  filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) %>%
  rowwise() %>% 
  mutate(treated = as.numeric(t_p_corr_running_e>0.5)) -> dat

# specify components of model
fe <- "iso + year"
cl <- "iso" # parametric SEs clustered by country
response_var <- "gr_pwt_frac"
tc <- "treated"
nlag <- 5

# DL formula
form_dl <- paste0(response_var," ~ e + e*",tc," + c + c*",tc)
for (j in c(1:nlag)){
  form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,
                    " + c_lag",j," + ","c_lag",j,"*",tc)  
}

mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)
xx <- cumulative_dl_coef(mdl,c("e:treated","treated:e_lag1","treated:e_lag2","treated:e_lag3","treated:e_lag4","treated:e_lag5"))


# extract cumulative coefficients
lags <- seq(0,nlag,by=1)
df <- data.frame("lag"=lags,
                 "beta"=numeric(length(lags)),
                 "se"=numeric(length(lags)),
                 "p"=numeric(length(lags)))

for (l in lags){
  dl_coefs <- c("e:treated")
  if (l>0){for (j in seq(1,l,by=1)){dl_coefs<-c(dl_coefs,paste0("treated:e_lag",j))}}
  fe_results<-cumulative_dl_coef(mdl,dl_coefs)
  df[df$lag==l,"beta"] <- fe_results[1]
  df[df$lag==l,"se"] <- fe_results[2]
  df[df$lag==l,"p"] <- fe_results[4]
}

write.csv(df,paste0(loc_save,"ENSO_discretized_yeareffects_test.csv"))




## what if both leads and lags are included?

# DL formula
form_dl <- paste0(response_var," ~ e + e*",tc," + c + c*",tc)
for (j in c(1:nlag)){
  form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc," + c_lag",j," + ","c_lag",j,"*",tc)  
  form_dl <- paste0(form_dl," + e_lead",j," + ","e_lead",j,"*",tc," + c_lead",j," + ","c_lead",j,"*",tc)  
}

mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)

# extract cumulative coefficients
lags <- seq(-5,5,by=1) #seq(0,nlag,by=1)
df <- data.frame("lag"=lags,
                 "beta"=numeric(length(lags)),
                 "se"=numeric(length(lags)),
                 "p"=numeric(length(lags)))

for (l in lags){
  dl_coefs <- c("treated:e_lead5")
  if (l > -5){
    for (j in seq(-4,l,by=1)){
      if (j < 0){dl_coefs <- c(dl_coefs,paste0("treated:e_lead",abs(j)))}
      else if (j==0){dl_coefs <- c(dl_coefs,"e:treated")}
      else if (j > 0){dl_coefs <- c(dl_coefs,paste0("treated:e_lag",abs(j)))}
    }
  }
  print(dl_coefs)
  fe_results <- cumulative_dl_coef(mdl,dl_coefs)
  df[df$lag==l,"beta"] <- fe_results[1]
  df[df$lag==l,"se"] <- fe_results[2]
  df[df$lag==l,"p"] <- fe_results[4]
}

write.csv(df,paste0(loc_save,"ENSO_discretized_yeareffects_test_leadlag.csv"))




## index of e times tau

min_gr <- -0.18
max_gr <- 0.18
fe <- "iso + year"
cl <- "iso"
response_var <- "gr_pwt_frac"
nlag <- 5
form_dl <- paste0(response_var," ~ etau + ctau")
for (j in c(1:nlag)){
  form_dl <- paste0(form_dl," + etau_lag",j," + ctau_lag",j)  
}

panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
dat %>% group_by(iso) %>% mutate(etau_sd = sd(etau,na.rm=T)) -> dat

mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)

# extract cumulative coefficients
lags <- seq(0,nlag,by=1)
df <- data.frame("lag"=lags,
                 "beta"=numeric(length(lags)),
                 "se"=numeric(length(lags)),
                 "p"=numeric(length(lags)))

for (l in lags){
  dl_coefs <- c("etau")
  if (l>0){for (j in seq(1,l,by=1)){dl_coefs<-c(dl_coefs,paste0("etau_lag",j))}}
  fe_results<-cumulative_dl_coef(mdl,dl_coefs)
  df[df$lag==l,"beta"] <- fe_results[1]
  df[df$lag==l,"se"] <- fe_results[2]
  df[df$lag==l,"p"] <- fe_results[4]
}

write.csv(df,paste0(loc_save,"ENSO_etau_yeareffects_test.csv"))


#####


##################  
#### Treatment effect homogeneity tests
##################


## effects in running 30-year blocks

min_gr <- -0.18
max_gr <- 0.18
fe <- "iso"
cl <- "iso"
response_var <- "gr_pwt_frac"
tc <- "t_p_corr_running"
nlag <- 5
form_dl <- paste0(response_var," ~ e + e*",tc,"_e + c + c*",tc,"_c")
for (j in c(1:nlag)){
  form_dl <- paste0(form_dl," + e_lag",j," + ","e_lag",j,"*",tc,"_e",
                    " + c_lag",j," + ","c_lag",j,"*",tc,"_c")  
}

panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr) -> dat
mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)
#print(summary(mdl))



coefs_sum <- c("e","e_lag1","e_lag2","e_lag3","e_lag4","e_lag5",
               "e:t_p_corr_running_e","t_p_corr_running_e:e_lag1",
               "t_p_corr_running_e:e_lag2","t_p_corr_running_e:e_lag3",
               "t_p_corr_running_e:e_lag4","t_p_corr_running_e:e_lag5")

orig_effect <- cumulative_dl_coef(mdl,coefs_sum)

yrs <- seq(1989,2019,by=1)
df <- data.frame("y1"=numeric(length(yrs)),
                 "y2"=numeric(length(yrs)),
                 "beta"=numeric(length(yrs)),
                 "se"=numeric(length(yrs)),
                 "p"=numeric(length(yrs)))

#running_coefs <- numeric(length(yrs))
for (yy in yrs){
  print(paste0(yy-30,"-",yy))
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,year<=yy,year>yy-30) -> dat
  mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)
  results <- cumulative_dl_coef(mdl,coefs_sum)
  df[yy-1988,"y1"] <- yy-30
  df[yy-1988,"y2"] <- yy
  df[yy-1988,"beta"] <- results[1]
  df[yy-1988,"se"] <- results[2]
  df[yy-1988,"p"] <- results[4]
  #running_coefs[yy-1998] <- results[1]
}

write.csv(df,paste0(loc_save,"ENSO_mainmodel_runningblocks_test.csv"))


## now leaving out countries
isos <- unique(panel$iso)

df <- data.frame("iso"=isos,
                 "beta"=numeric(length(isos)),
                 "se"=numeric(length(isos)),
                 "p"=numeric(length(isos)))

for (i in isos){
  print(i)
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,iso!=i) -> dat
  mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)
  results <- cumulative_dl_coef(mdl,coefs_sum)
  df[df$iso==i,"beta"] <- results[1]
  df[df$iso==i,"se"] <- results[2]
  df[df$iso==i,"p"] <- results[4]
}

write.csv(df,paste0(loc_save,"ENSO_mainmodel_excludecountries_test.csv"))


## now leaving out years
yrs_exclude <- unique(panel$year)
df <- data.frame("year"=yrs_exclude,
                 "beta"=numeric(length(yrs_exclude)),
                 "se"=numeric(length(yrs_exclude)),
                 "p"=numeric(length(yrs_exclude)))

for (y in yrs_exclude){
  print(y)
  panel %>% filter(gr_pwt_frac<max_gr,gr_pwt_frac>min_gr,year!=y) -> dat
  mdl <- felm(as.formula(paste0(form_dl," | ",fe," | 0 | ",cl)),data=dat)
  results <- cumulative_dl_coef(mdl,coefs_sum)
  df[df$year==y,"beta"] <- results[1]
  df[df$year==y,"se"] <- results[2]
  df[df$year==y,"p"] <- results[4]
}

write.csv(df,paste0(loc_save,"ENSO_mainmodel_excludeyears_test.csv"))


####







