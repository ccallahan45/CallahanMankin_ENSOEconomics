# Growth effects of changes in ENSO and its teleconnections
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

#### Mechanics
# Dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from rasterio import features
from affine import Affine
import geopandas as gp
import descartes
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.formula.api import ols as reg
from functools import reduce

# Data location

loc_e = "../Data/ENSO_Indices/CMIP6/"
loc_e_obs = "../Data/ENSO_Indices/"
loc_shp = "../Data/ProcessedCountryShapefile/"
loc_panel = "../Data/Panel/"
loc_ssp_gdp = "../Data/SSP/GDP/"
loc_ssp_pop = "../Data/SSP/population/"
loc_coefs = "../Data/RegressionResults/"
loc_tc_cmip6 = "../Data/Teleconnections/CMIP6/"
loc_tc_obs = "../Data/Teleconnections/"
loc_out = "../Data/Damages/"

# warnings
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in .resample")

# yearly mean functio
def monthly_to_yearly_mean(x):

        # calculate annual mean from monthly data
        # after weighting for the difference in month length
        # x must be data-array with time coord
        # xarray must be installed

        # x_yr = x.resample(time="YS").mean(dim="time") is wrong
        # because it doesn't weight for the # of days in each month

        days_in_mon = x.time.dt.days_in_month
        wgts = days_in_mon.groupby("time.year")/days_in_mon.groupby("time.year").sum()
        ones = xr.where(x.isnull(),0.0,1.0)
        x_sum = (x*wgts).resample(time="YS").sum(dim="time")
        ones_out = (ones*wgts).resample(time="YS").sum(dim="time")
        return(x_sum/ones_out)

# Years

y1_ssp = 2020
y2_ssp = 2100
years_ssp = np.arange(y1_ssp,y2_ssp+1,1)
y1_hist = 1940
y2_hist = 2019
y1_damages = 2020
y2_damages = 2099
years_damages = np.arange(y1_damages,y2_damages+1,1)
y1_cmip_e = 1850
y2_cmip_e = 2099
y1_cmip_tc = 1940
y2_cmip_tc = 2099

# Shapefile
shp = gp.read_file(loc_shp)
iso_shp = shp.ISO3.values

## establish level of damage persistence
#persistence = # either "partial" or "full"

## get coefficients
response = "gr_pwt_frac" # penn world tables growth
enso_var_e = "e_e-and-c" # e-index
enso_var_c = "c_e-and-c" # c-index
trend = "none" # no controls for trends
nlag_orig = 5 # cumulative lag length
lags_orig = np.arange(0,nlag_orig+1,1)

# ENSO coefficients
coefs_main_e_name = "ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag"+str(nlag_orig)+"_"+enso_var_e+"_"+response+"_trend"+trend+".csv"
coefs_main_e_df = pd.read_csv(loc_coefs+coefs_main_e_name,index_col=0)
boot = coefs_main_e_df.boot.values
nboot = len(boot)
coefs_main_e_orig = xr.DataArray(coefs_main_e_df.iloc[:,1:].values,coords=[boot,lags_orig],dims=["boot","lag"])

coefs_main_c_name = "ENSO_teleconnection-interacted_coefs_bootstrap-dist_lag"+str(nlag_orig)+"_"+enso_var_c+"_"+response+"_trend"+trend+".csv"
coefs_main_c_df = pd.read_csv(loc_coefs+coefs_main_c_name,index_col=0)
coefs_main_c_orig = xr.DataArray(coefs_main_c_df.iloc[:,1:].values,coords=[boot,lags_orig],dims=["boot","lag"])

# interaction coefficients
coefs_int_e_name = "ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag"+str(nlag_orig)+"_"+enso_var_e+"_"+response+"_trend"+trend+".csv"
coefs_int_e_df = pd.read_csv(loc_coefs+coefs_int_e_name,index_col=0)
coefs_int_e_orig = xr.DataArray(coefs_int_e_df.iloc[:,1:].values,coords=[boot,lags_orig],dims=["boot","lag"])

coefs_int_c_name = "ENSO_teleconnection-interaction_coefs_bootstrap-dist_lag"+str(nlag_orig)+"_"+enso_var_c+"_"+response+"_trend"+trend+".csv"
coefs_int_c_df = pd.read_csv(loc_coefs+coefs_int_c_name,index_col=0)
coefs_int_c_orig = xr.DataArray(coefs_int_c_df.iloc[:,1:].values,coords=[boot,lags_orig],dims=["boot","lag"])

# level of persistence of damages
persist = "full" # "partial" or "full"
# if partial, we want to put in a rebound back to zero
if persist=="full":
    # full persistence, no rebound
    coefs_main_e = coefs_main_e_orig*1.0
    coefs_main_c = coefs_main_c_orig*1.0
    coefs_int_e = coefs_int_e_orig*1.0
    coefs_int_c = coefs_int_c_orig*1.0
    nlag = int(nlag_orig*1.0)
    lags = np.arange(0,nlag+1,1)
elif persist=="partial":
    # rebound to prevent damages from lasting 4ever
    nlag = 14 # years after event 
    lags = np.arange(0,nlag+1,1)
    coefs_main_e = xr.DataArray(np.full((nboot,nlag+1),np.nan),coords=[boot,lags],dims=["boot","lag"])
    coefs_main_e.loc[:,0:nlag_orig] = coefs_main_e_orig
    # zero effect in the middle -- plateau period
    # then a reverse effect/rebound from years 10 to 15
    coefs_main_e.loc[:,(nlag-nlag_orig):nlag] = coefs_main_e_orig[:,::-1].values*-1

    coefs_int_e = xr.DataArray(np.full((nboot,nlag+1),np.nan),coords=[boot,lags],dims=["boot","lag"])
    coefs_int_e.loc[:,0:nlag_orig] = coefs_int_e_orig
    coefs_int_e.loc[:,(nlag-nlag_orig):nlag] = coefs_int_e_orig[:,::-1].values*-1

    ## figure out c-index persistence
    coefs_main_c = xr.DataArray(np.full((nboot,nlag+1),np.nan),coords=[boot,lags],dims=["boot","lag"])
    coefs_main_c.loc[:,0:nlag_orig] = coefs_main_c_orig
    coefs_main_c.loc[:,(nlag-nlag_orig):nlag] = coefs_main_c_orig[:,::-1].values*-1

    coefs_int_c = xr.DataArray(np.full((nboot,nlag+1),np.nan),coords=[boot,lags],dims=["boot","lag"])
    coefs_int_c.loc[:,0:nlag_orig] = coefs_int_c_orig
    coefs_int_c.loc[:,(nlag-nlag_orig):nlag] = coefs_int_c_orig[:,::-1].values*-1

else:
    print("invalid persistence flag",flush=True)
    sys.exit()


# Observed teleconnections
y1_tc = 1960
y2_tc = 2019
tc_metric_e = "combined_corr_e_running"
tc_metric_c = "combined_corr_c_running"
tc_ds = xr.open_dataset(loc_tc_obs+"ENSO_observed_teleconnections_DJF_"+str(y1_tc)+"-"+str(y2_tc)+".nc")
tc_e_in = tc_ds.data_vars[tc_metric_e]
obs_tc_e = tc_e_in.where(tc_e_in>0,np.nan) # can only be positive, set nans
tc_c_in = tc_ds.data_vars[tc_metric_c]
obs_tc_c = tc_c_in.where(tc_c_in>0,np.nan) # can only be positive, set nans


## get basic SSP data
ssp_gdp_in = pd.read_csv(loc_ssp_gdp+"SSP_GDP.csv")
ssp_pop_in = pd.read_csv(loc_ssp_pop+"SSP_Population.csv")

countries = np.array([x for x in np.unique(sorted(ssp_gdp_in.Region.values)) if x in obs_tc_e.iso.values])
countries_pop = np.array([x for x in np.unique(sorted(ssp_pop_in.Region.values)) if x in obs_tc_e.iso.values])


## experiments
experiments = ["ssp126","ssp245","ssp370","ssp585"]
#experiments = ["ssp370","ssp245"]

for e in experiments:

    print(e,flush=True)

    print("reading and processing SSP data",flush=True)
    sspname = e[0:4].upper()

    # for each experiment, there is a corresponding SSP

    ssp_gdp = xr.DataArray(np.full((len(years_ssp),len(countries)),np.nan),
                           coords=[years_ssp,countries],
                           dims=["time","iso"])
    ssp_pop = xr.DataArray(np.full((len(years_ssp),len(countries)),np.nan),
                           coords=[years_ssp,countries],
                           dims=["time","iso"])

    yrs_gdp = np.arange(y1_ssp,y2_ssp+5,5)

    for i in np.arange(0,len(countries),1):
        iso_gdp = countries[i]
        indices = (ssp_gdp_in.Region==iso_gdp)&(ssp_gdp_in.Scenario==sspname)
        if iso_gdp in countries:
            iso_in = ssp_gdp_in.loc[indices,str(y1_ssp):str(y2_ssp)].values[0]
            iso_in_xr = xr.DataArray(iso_in,coords=[yrs_gdp],dims=["time"])
            iso_in_interp = iso_in_xr.interp(time=years_ssp,method="linear")
            ssp_gdp.loc[:,iso_gdp] = iso_in_interp.values*(1e9)


        indices_pop = (ssp_pop_in.Region==iso_gdp)&(ssp_pop_in.Scenario==sspname)&(ssp_pop_in.Model=="IIASA-WiC POP")
        if iso_gdp in countries_pop:
            iso_in_pop = ssp_pop_in.loc[indices_pop,str(y1_ssp):str(y2_ssp)].values[0]
            iso_pop_xr = xr.DataArray(iso_in_pop,coords=[yrs_gdp],dims=["time"])
            iso_pop_interp = iso_pop_xr.interp(time=years_ssp,method="linear")
            ssp_pop.loc[:,iso_gdp] = iso_pop_interp.values*(1e6)

    # GDP per capita
    ssp_gpc = ssp_gdp/ssp_pop

    # growth as fractional change, not difference of logs
    #ssp_lngpc = np.log(ssp_gpc)
    ssp_growth1 = ssp_gpc.diff(dim="time",n=1) #ssp_lngpc.diff(dim="time",n=1)
    ssp_frac_growth = ssp_growth1/(ssp_gpc.loc[:2099,:].values)

    growth_nans = xr.DataArray(np.full((1,len(countries)),np.nan),
                               coords=[[y1_damages],countries],
                               dims=["time","iso"])
    ssp_growth = xr.concat([growth_nans,ssp_frac_growth],dim="time")

    ssp_gpc = ssp_gpc.transpose("iso","time").loc[:,years_damages]
    ssp_pop = ssp_pop.transpose("iso","time").loc[:,years_damages]
    ssp_gdp = ssp_gdp.transpose("iso","time").loc[:,years_damages]
    ssp_growth = ssp_growth.transpose("iso","time").loc[:,years_damages]

    print("getting model data for ENSO indices and teleconnections",flush=True)
    # now we want to get the list of models that have data
    # for both enso and teleconnections
    e_models = np.array([x for x in sorted(os.listdir(loc_e)) if ("historical-"+e in x)&(x.endswith(".nc"))])
    e_models_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in e_models])

    tc_models = np.array([x for x in sorted(os.listdir(loc_tc_cmip6)) if ("historical-"+e in x)&(x.endswith(".nc"))])
    tc_models_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in tc_models])

    # intersection of models with ENSO data and teleconnections data
    models_intersect = reduce(np.intersect1d,(e_models_prefix,tc_models_prefix))

    # read datasets
    models_e_list = np.array([loc_e+x+"_historical-"+e+"_ENSO_indices_"+str(y1_cmip_e)+"-"+str(y2_cmip_e)+".nc" for x in models_intersect])
    models_tc_list = np.array([loc_tc_cmip6+x+"_historical-"+e+"_country_teleconnections_"+str(y1_cmip_tc)+"-"+str(y2_cmip_tc)+".nc" for x in models_intersect])
    e_model_ds = xr.open_mfdataset(models_e_list,combine="nested",concat_dim="model")
    e_model_ds.coords["model"] = models_intersect
    tc_model_ds = xr.open_mfdataset(models_tc_list,combine="nested",concat_dim="model")
    tc_model_ds.coords["model"] = models_intersect

    # get alpha and limit models
    alpha_threshold = -0.17 # half the obs value of -0.34
    alpha = e_model_ds.alpha.load()
    models_final = models_intersect[alpha.values<alpha_threshold]
    alpha_final = alpha[alpha.values<alpha_threshold]

    # calculate e- and c-index amplitude change
    e_index = e_model_ds.e_index.load().loc[models_final,:].shift(time=1)
    c_index = e_model_ds.c_index.load().loc[models_final,:].shift(time=1)
    #e_djf = e_index[:,e_index.time.dt.month<=3].resample(time="YS").mean(dim="time")
    #c_djf = c_index[:,c_index.time.dt.month<=3].resample(time="YS").mean(dim="time")
    e_djf = monthly_to_yearly_mean(e_index[:,e_index.time.dt.month<=3])
    c_djf = monthly_to_yearly_mean(c_index[:,c_index.time.dt.month<=3])
    e_djf.coords["time"] = e_djf.time.dt.year.values
    c_djf.coords["time"] = c_djf.time.dt.year.values

    e_hist = e_djf.loc[:,y1_hist:y2_hist]
    e_future = e_djf.loc[:,y1_damages:y2_damages]
    c_hist = c_djf.loc[:,y1_hist:y2_hist]
    c_future = c_djf.loc[:,y1_damages:y2_damages]

    # ratio of amplitudes
    e_amp_ratio = e_future.std(dim="time")/e_hist.std(dim="time") - 1
    c_amp_ratio = c_future.std(dim="time")/c_hist.std(dim="time") - 1

    # center time series and reset amplitude to historical level
    e_future_centered = e_future*1.0 #- e_future.mean(dim="time")
    c_future_centered = c_future*1.0 #- c_future.mean(dim="time")
    e_future_hist_amp = e_future_centered*(1 - e_amp_ratio)
    c_future_hist_amp = c_future_centered*(1 - c_amp_ratio)

    # our comparison will be e_future_centered and e_future_hist_amp, same for c-index
    # to compare the same time series and isolate the effect of changing amplitude

    # now calculate observed and counterfactual teleconnections
    tc_model_e = tc_model_ds.data_vars["teleconnections_corr_running"].load().loc[models_final,"e",:,countries]
    tc_model_c = tc_model_ds.data_vars["teleconnections_corr_running"].load().loc[models_final,"c",:,countries]

    # delta-method -- add change to historical values to implicitly bias correct
    tc_e_change = tc_model_e.loc[:,"future",:] - tc_model_e.loc[:,"hist",:]
    tc_c_change = tc_model_c.loc[:,"future",:] - tc_model_c.loc[:,"hist",:]

    tclimit = False
    if tclimit:
        tclim = "_tclimit_"
    else:
        tclim = "_"
    tc_e_future = obs_tc_e + tc_e_change
    tc_c_future = obs_tc_c + tc_c_change
    if tclimit:
        tc_e_future = tc_e_future.where(tc_e_future<=np.nanmax(obs_tc_e.values),np.nanmax(obs_tc_e.values))
        tc_c_future = tc_c_future.where(tc_c_future<=np.nanmax(obs_tc_c.values),np.nanmax(obs_tc_c.values))
        tc_e_future = tc_e_future.where(tc_e_future>=np.nanmin(obs_tc_e.values),np.nanmin(obs_tc_e.values))
        tc_c_future = tc_c_future.where(tc_c_future>=np.nanmin(obs_tc_c.values),np.nanmin(obs_tc_c.values))
    tc_e_hist = obs_tc_e.expand_dims(model=tc_e_future.model)
    tc_c_hist = obs_tc_c.expand_dims(model=tc_c_future.model)

    print("calculating change in growth",flush=True)

    # expand coefficients -- country x boot x lag
    coefs_main_e_xr = coefs_main_e.expand_dims(iso=countries)
    coefs_int_e_xr = coefs_int_e.expand_dims(iso=countries)
    coefs_main_c_xr = coefs_main_c.expand_dims(iso=countries)
    coefs_int_c_xr = coefs_int_c.expand_dims(iso=countries)

    # calculate marginal effects
    #me_e_tcchange = coefs_e_xr
    me_e_tcfuture = coefs_main_e_xr + coefs_int_e_xr*tc_e_future
    me_e_tchist = coefs_main_e_xr + coefs_int_e_xr*tc_e_hist
    me_c_tcfuture = coefs_main_c_xr + coefs_int_c_xr*tc_c_future
    me_c_tchist = coefs_main_c_xr + coefs_int_c_xr*tc_c_hist

    # set up final dataarray
    #components = ["e_both","e_amp","e_tc","c_both"] #,"c_amp","c_tc"]
    indices = ["E","C"] #,"E"]
    sensitivity = ["main","notcchange"] # main projections, no la nina effect, no teleconnection change
    for index in indices:
        print(index,flush=True)
        for s in sensitivity:
            print(s,flush=True)

            if index=="E":
                if s=="main":
                    me_future = me_e_tcfuture
                    enso_future = e_future_centered
                    me_hist = me_e_tchist
                    enso_hist = e_future_hist_amp
                elif s=="notcchange":
                    me_future = me_e_tcfuture
                    enso_future = e_future_centered
                    me_hist = me_e_tcfuture*1.0
                    enso_hist = e_future_hist_amp
            elif index=="C":
                if s=="main":
                    me_future = me_c_tcfuture
                    enso_future = c_future_centered
                    me_hist = me_c_tchist
                    enso_hist = c_future_hist_amp
                elif s=="notcchange":
                    me_future = me_c_tcfuture
                    enso_future = c_future_centered
                    me_hist = me_c_tcfuture*1.0
                    enso_hist = c_future_hist_amp

            # significance
            #me_lower = me_future.sum(dim="lag").quantile(0.025,dim="boot")
            #me_upper = me_future.sum(dim="lag").quantile(0.975,dim="boot")
            #me_significance[cc,:,:] = (((me_lower < 0)&(me_upper < 0)).astype(int)).transpose("model","iso")

            # change in growth
            dg = me_future.sel(lag=0)*enso_future - me_hist.sel(lag=0)*enso_hist
            for l in np.arange(1,nlag+1,1):
                print(l)
                delta = me_future.sel(lag=l)*enso_future.shift(time=l) - me_hist.sel(lag=l)*enso_hist.shift(time=l)
                dg = dg + delta.where(~np.isnan(delta),0.0)

            # now calculate damages
            print("calculating counterfactual GDP",flush=True)
            def calc_counterfactual_gdp(orig_growth,orig_gpc,delta_growth,dimlist):
                dimlist2 = dimlist[:-1]
                yrs = orig_growth.coords["time"].values
                cf_growth = (orig_growth.transpose("time","iso")+delta_growth).transpose(*dimlist)
                cf_gpc = xr.DataArray(np.full(delta_growth.shape,np.nan),
                                        coords=delta_growth.coords,dims=dimlist)
                for yy in np.arange(1,len(yrs),1):
                    if yy == 1:
                        cf_gdp_year = orig_gpc.sel(time=yrs[yy-1]).drop("time")
                    else:
                        cf_gdp_year = cf_gpc.sel(time=yrs[yy-1]).drop("time")
                    cf_gpc.loc[dict(time=yrs[yy])] = (cf_gdp_year+(cf_growth.loc[dict(time=yrs[yy])]*cf_gdp_year)).transpose(*dimlist2)
                return([cf_gpc,cf_growth])

            dimlist = ["boot","model","iso","time"]
            cf_gpc, cf_gr = calc_counterfactual_gdp(ssp_growth,ssp_gpc,dg.transpose(*dimlist),dimlist)

            print("calculating final numbers and writing out",flush=True)

            ## calculate damages/GDP change
            # gdp per capita in percent terms
            gpc_change_pct = 100*(cf_gpc - ssp_gpc)/ssp_gpc
            # total GDP
            gdp_change = cf_gpc*ssp_pop - ssp_gdp

            ## summarize and write out
            gpc_change_pct_mean = gpc_change_pct.mean(dim=["model","boot"]).transpose("iso","time")
            gpc_change_pct_median = gpc_change_pct.mean(dim="time").median(dim=["model","boot"])
            gdp_change_mean = gdp_change.mean(dim=["model","boot"]).transpose("iso","time")

            qs = [0.005,0.025,0.05,0.25,0.5,0.75,0.95,0.975,0.995]
            gdp_change_quantiles = gdp_change.quantile(qs,dim=["model","boot"]).transpose("quantile","iso","time")
            gdp_change_global = gdp_change.sum(dim="iso")
            dimlist_out = ["iso","time","boot","model"]
            #print(gdp_change_global.sum(dim="time").mean(dim=["boot","model"]))
            #print(gdp_change_global.sum(dim="time").median(dim=["boot","model"]))

            # time mean
            gdp_change_pct_overall_mean = gpc_change_pct.mean(dim="time").mean(dim=["model","boot"])
            gdp_change_pct_overall_sd = gpc_change_pct.mean(dim="time").std(dim=["model","boot"])

            cf_gr_out = cf_gr.mean(dim="iso")
            damage_ds = xr.Dataset({"gpc_change_pct":(dimlist_out,gpc_change_pct.transpose(*dimlist_out)),
                                    "gpc_change_pct_mean":(["iso","time"],gpc_change_pct_mean),
                                    "gpc_change_pct_median":(["iso"],gpc_change_pct_median),
                                    "gpc_change_pct_overall_mean":(["iso"],gdp_change_pct_overall_mean),
                                    "gpc_change_pct_overall_sd":(["iso"],gdp_change_pct_overall_sd),
                                    "gdp_change":(dimlist_out,gdp_change.transpose(*dimlist_out)),
                                    "gdp_change_mean":(["iso","time"],gdp_change_mean),
                                    "gdp_change_global":(["time","boot","model"],gdp_change_global.transpose("time","boot","model")),
                                    "ssp_gdp":(["iso","time"],ssp_gdp),
                                    "ssp_gpc":(["iso","time"],ssp_gpc),
                                    "ssp_growth":(["iso","time"],ssp_growth),
                                    "cf_growth":(["time","boot","model"],cf_gr_out.transpose("time","boot","model")),
                                    "alpha":(["model"],alpha_final)},
                                    coords={"iso":(["iso"],countries),
                                            "time":(["time"],years_damages),
                                            "model":(["model"],models_final),
                                            "boot":(["boot"],boot),
                                            "quantile":(["quantile"],qs)})

            damage_ds.attrs["creation_date"] = str(datetime.datetime.now())
            damage_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
            damage_ds.attrs["variable_description"] = "country-level economic damages/changes from future changes in ENSO amplitude and teleconnections"
            damage_ds.attrs["created_from"] = os.getcwd()+"/ENSO_Future_Damages.py"

            fname_out = loc_out+"CMIP6_"+index+"index_damages_amplitude_teleconnections_"+e+"_"+s+"_"+persist+"persistence_"+str(y1_damages)+"-"+str(y2_damages)+".nc"
            damage_ds.to_netcdf(fname_out,mode="w")
            print(fname_out,flush=True)
