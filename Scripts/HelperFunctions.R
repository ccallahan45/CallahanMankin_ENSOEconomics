
add_lag <- function(df,col,nlag){
  colname <- paste(col,"_lag",nlag,sep="")
  df[colname] <- numeric(dim(df)[1])
  yr_unique <- unique(df$year)
  cnum_unique <- unique(df$countrynum) 
  for (yy in c(1:length(yr_unique))){
    for (cc in c(1:length(cnum_unique))){
      yr = yr_unique[yy]
      c = cnum_unique[cc]
      index <- (df$year==yr)&(df$countrynum==c)
      if (yr <= (min(yr_unique) + (nlag-1))){ df[index,colname] <- NA } else {
        df[index,colname] <- df[(df$year==yr-nlag)&(df$countrynum==c),col]
      }
    }
  }
  return(df)
}
add_lead <- function(df,col,nlead){
  colname <- paste(col,"_lead",nlead,sep="")
  df[colname] <- numeric(dim(df)[1])
  yr_unique <- unique(df$year)
  cnum_unique <- unique(df$countrynum) 
  for (yy in c(1:length(yr_unique))){
    for (cc in c(1:length(cnum_unique))){
      yr = yr_unique[yy]
      c = cnum_unique[cc]
      index <- (df$year==yr)&(df$countrynum==c)
      if (yr >= (max(yr_unique) - (nlead-1))){ df[index,colname] <- NA } else {
        df[index,colname] <- df[(df$year==yr+nlead)&(df$countrynum==c),col]
      }
    }
  }
  return(df)
}




custom_theme_leg <- function(xpos,ypos){
  theme_classic() + 
    theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),
          axis.title.y=element_text(margin=margin(t=0,r=10,b=0,l=0)),
          axis.title.x=element_text(margin=margin(t=10,r=0,b=0,l=0)),
          plot.title=element_text(color="black",size=16,hjust=0.5,face="bold"),
          axis.line = element_line(color = 'black',size = 0.35),
          axis.ticks = element_line(colour = "black",size = 0.35),
          axis.ticks.length=unit(0.3,"cm"),
          axis.text.x=element_text(color="black"),
          axis.text.y=element_text(color="black"),
          legend.position=c(xpos,ypos),legend.text=element_text(size=10),
          legend.title=element_text(size=10,face="bold"),
          legend.background=element_rect(fill=rgb(0.98,0.98,0.98),
                                         size=0.5,linetype="solid",
                                         colour="lightgray"))
}

custom_theme_noleg <- function(){
  theme_classic() + 
    theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),
          axis.title.y=element_text(margin=margin(t=0,r=10,b=0,l=0)),
          axis.title.x=element_text(margin=margin(t=10,r=0,b=0,l=0)),
          plot.title=element_text(color="black",size=16,hjust=0.5,face="bold"),
          axis.line = element_line(color = 'black',size = 0.35),
          axis.ticks = element_line(colour = "black",size = 0.35),
          axis.ticks.length=unit(0.3,"cm"),
          axis.text.x=element_text(color="black"),
          axis.text.y=element_text(color="black"))
  #axis.text.x.bottom = element_text(vjust=0)
}

custom_theme_noleg_v2 <- function(){
  theme_classic() + 
    theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),
          axis.title.y=element_text(margin=margin(t=0,r=10,b=0,l=0)),
          axis.title.x=element_text(margin=margin(t=10,r=0,b=0,l=0)),
          plot.title=element_text(color="black",size=16,hjust=0.5,face="bold"),
          axis.line = element_line(color = 'black',size = 0.35),
          axis.ticks = element_line(colour = "black",size = 0.35),
          axis.ticks.length=unit(0.3,"cm"),
          axis.text.x=element_text(color="black"),
          axis.text.y=element_text(color="black")) + 
    theme(plot.margin=unit(c(0.5,4,0.4,3),"cm")) +
    background_grid(major="none",minor="none")
}

quadratic_detrend <- function(x){
  time <- c(1:length(x))
  time2 <- time**2
  intercept <- coef(summary(lm(x ~ time + time2)))[1,1]
  lin_coef <- coef(summary(lm(x ~ time + time2)))[2,1]
  quad_coef <- coef(summary(lm(x ~ time + time2)))[3,1]
  yhat <- intercept + (lin_coef * time) + (quad_coef * time2)
  x_detrend <- x - yhat
  return(x_detrend)
}

