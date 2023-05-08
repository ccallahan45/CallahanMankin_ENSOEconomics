# Historical and future ENSO teleconnections in CMIP6 models
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.formula.api import ols as reg
from functools import reduce

# locations
loc_temp = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/CountryTemp/CMIP6/"
loc_precip = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/CountryPrecip/CMIP6/"
loc_enso = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/ENSO_Indices/CMIP6/"
loc_out = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/Teleconnections/CMIP6/"

# partial correlation function
def partial_correlation(x,y,z):
    from scipy import stats
    from statsmodels.formula.api import ols as reg
    df = pd.DataFrame({"x":x,"y":y,"z":z})
    x_z_resids = reg("x~z",data=df).fit().resid.values
    y_z_resids = reg("y~z",data=df).fit().resid.values
    corr, p = stats.pearsonr(x_z_resids,y_z_resids)
    #return([corr,p])
    return(corr)

# anomaly function
def calc_anom_iso_time(x,y1,y2):
    # standardized anomalies for
    # data in iso x time format
    # climatology period from y1-y2
    clm_mean = x.loc[:,str(y1)+"-01-01":str(y2)+"-12-31"].groupby("time.month").mean("time")
    clm_sd = x.loc[:,str(y1)+"-01-01":str(y2)+"-12-31"].groupby("time.month").std("time")
    anom = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        x.groupby("time.month"),
        clm_mean,clm_sd,
    )
    return(anom)

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


# experiments and other info
experiments = ["ssp126","ssp245","ssp370","ssp585"]
y1_in = 1850
y2_in = 2099
y1_hist = 1940
y2_hist = 2019
y1_future = 2020
y2_future = 2099 # the two periods need to be the same length
#periods = [str(y1_hist)+"-"+str(y2_hist),str(y1_future)+"-"+str(y2_future)]
periods = ["hist","future"]
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in")

# loop through models
for e in experiments:
    print(e,flush=True)

    # models with ENSO data
    enso_models = np.array([x for x in sorted(os.listdir(loc_enso)) if ("historical-"+e in x)&(x.endswith(".nc"))])
    enso_models_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in enso_models])

    # models with temp data
    temp_models = np.array([x for x in sorted(os.listdir(loc_temp)) if ("historical-"+e in x)&(x.endswith(".nc"))])
    temp_models_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in temp_models])

    # models with precip data
    precip_models = np.array([x for x in sorted(os.listdir(loc_precip)) if ("historical-"+e in x)&(x.endswith(".nc"))])
    precip_models_prefix = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in precip_models])

    # intersection of all three
    models = reduce(np.intersect1d,(enso_models_prefix,temp_models_prefix,precip_models_prefix))
    print(models)

    #models_limit = ["ACCESS-CM2","ACCESS-ESM1-5","CanESM5","CESM2","CESM2-WACCM","FGOALS-g3","HadGEM3-GC31-LL","HadGEM3-GC31-MM","MIROC-ES2L","IPSL-CM6A-LR","KACE-1-0-G"]
    models_limit = []
    # loop through models
    for m in models:
        mname = m.split("_")[0]
        mreal = m.split("_")[1]
        mfull = "pr_day_"+mname

        print(m,flush=True)
        if (mname in models_limit):
            continue
        #if (e=="ssp585")&(mname=="MIROC6")&(int(mreal[1])<=3):
        #    continue

        # read in ENSO
        enso_monthly = xr.open_dataset(loc_enso+m+"_historical-"+e+"_ENSO_indices_"+str(y1_in)+"-"+str(y2_in)+".nc")
        e_monthly = enso_monthly.e_index.loc[str(y1_hist)+"-01-01":str(y2_future)+"-12-31"]
        eshift = e_monthly.shift(time=1)
        #e_djf = eshift[eshift.time.dt.month<=3].resample(time="YS").mean(dim="time")
        e_djf = monthly_to_yearly_mean(eshift[eshift.time.dt.month<=3])
        c_monthly = enso_monthly.c_index.loc[str(y1_hist)+"-01-01":str(y2_future)+"-12-31"]
        cshift = c_monthly.shift(time=1)
        #c_djf = cshift[cshift.time.dt.month<=3].resample(time="YS").mean(dim="time")
        c_djf = monthly_to_yearly_mean(cshift[cshift.time.dt.month<=3])

        # read in standardized country temp and precip
        t_ds = xr.open_dataset(loc_temp+m+"_historical-"+e+"_country_temperature_monthly_"+str(y1_in)+"-"+str(y2_in)+".nc")
        t_abs = t_ds.temp.loc[:,str(y1_hist)+"-01-01":str(y2_future)+"-12-31"]
        p_ds = xr.open_dataset(loc_precip+m+"_historical-"+e+"_country_precipitation_monthly_"+str(y1_in)+"-"+str(y2_in)+".nc")
        p_abs = p_ds.precip.loc[:,str(y1_hist)+"-01-01":str(y2_future)+"-12-31"]

        iso = p_abs.coords["iso"]

        indices = ["e","c"]
        # create final teleconnection variables
        tc_corr = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_corr_t = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_corr_p = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_reg = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_reg_t = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_reg_p = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_corr_running = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_corr_running_t = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_corr_running_p = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_reg_running = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_reg_running_t = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])
        tc_reg_running_p = xr.DataArray(np.full((len(indices),len(periods),len(iso)),np.nan),
                                coords=[indices,periods,iso],dims=["index","period","iso"])

        # loop through indices and periods
        for index in indices:
            if index=="e":
                enso = e_djf
            elif index=="c":
                enso = c_djf
            print(index)

            for p in periods:
                if p=="hist":
                    y1_period = y1_hist
                    y2_period = y2_hist
                elif p=="future":
                    y1_period = y1_future
                    y2_period = y2_future
                print(p+", "+str(y1_period)+"-"+str(y2_period))

                enso_period = enso.loc[str(y1_period)+"-01-01":str(y2_period)+"-12-31"]
                t_period_abs = t_abs.loc[:,str(y1_period)+"-01-01":str(y2_period)+"-12-31"]
                p_period_abs = p_abs.loc[:,str(y1_period)+"-01-01":str(y2_period)+"-12-31"]
                t_period = calc_anom_iso_time(t_period_abs,y1_period,y2_period)
                p_period = calc_anom_iso_time(p_period_abs,y1_period,y2_period)

                # now loop through countries
                # then loop through month and identify month of peak teleconnection
                #mns = np.arange(-6,12+1,1)
                # june through august -- summer season in both directions
                mns = [-6,-7,-8,-9,-10,-11,-12,1,2,3,4,5,6,7,8] #-5 ,7,8,9,10,11,12]
                for i in iso:
                    t_i = t_period.loc[i,:]
                    p_i = p_period.loc[i,:]
                    if (~np.any(np.isnan(t_i.values)))&(~np.any(np.isnan(p_i.values)))&(~np.any(np.isinf(t_i.values)))&(~np.any(np.isinf(p_i.values))):
                        t_coefs = xr.DataArray(np.zeros(len(mns)),coords=[mns],dims=["month"])
                        p_coefs = xr.DataArray(np.zeros(len(mns)),coords=[mns],dims=["month"])
                        t_regcoefs = xr.DataArray(np.zeros(len(mns)),coords=[mns],dims=["month"])
                        p_regcoefs = xr.DataArray(np.zeros(len(mns)),coords=[mns],dims=["month"])

                        for mm in np.arange(0,len(mns),1):
                            if mns[mm]<0:
                                t = signal.detrend(t_i[t_i.time.dt.month==np.abs(mns[mm])].shift(time=1)[1:].values,type="linear")
                                pr = signal.detrend(p_i[p_i.time.dt.month==np.abs(mns[mm])].shift(time=1)[1:].values,type="linear")
                                enso_final = enso_period[1:]
                            else:
                                t = signal.detrend(t_i[t_i.time.dt.month==mns[mm]][1:].values,type="linear")
                                pr = signal.detrend(p_i[p_i.time.dt.month==mns[mm]][1:].values,type="linear")
                                enso_final = enso_period[1:]
                            t_coefs[mm] = partial_correlation(enso_final.values,t,pr)
                            p_coefs[mm] = partial_correlation(enso_final.values,pr,t)
                            df = pd.DataFrame({"t":t,"p":pr,"enso":enso_final.values})
                            t_regcoefs[mm] = reg("t ~ enso + p",data=df).fit().params["enso"]
                            p_regcoefs[mm] = reg("p ~ enso + t",data=df).fit().params["enso"]

                        t_corr = np.amax(np.abs(t_coefs))
                        p_corr = np.amax(np.abs(p_coefs))
                        t_reg = np.amax(np.abs(t_regcoefs))
                        p_reg = np.amax(np.abs(p_regcoefs))
                        t_corr_running = np.amax(np.abs(t_coefs).rolling(month=3).mean())
                        p_corr_running = np.amax(np.abs(p_coefs).rolling(month=3).mean())
                        t_reg_running = np.amax(np.abs(t_regcoefs).rolling(month=3).mean())
                        p_reg_running = np.amax(np.abs(p_regcoefs).rolling(month=3).mean())
                        tc_corr_t.loc[index,p,i] = t_corr.values
                        tc_corr_p.loc[index,p,i] = p_corr.values
                        tc_corr.loc[index,p,i] = t_corr.values + p_corr.values
                        tc_reg_t.loc[index,p,i] = t_reg.values
                        tc_reg_p.loc[index,p,i] = p_reg.values
                        tc_reg.loc[index,p,i] = t_reg.values + p_reg.values
                        tc_corr_running.loc[index,p,i] = t_corr_running.values + p_corr_running.values
                        tc_corr_running_t.loc[index,p,i] = t_corr_running.values
                        tc_corr_running_p.loc[index,p,i] = p_corr_running.values
                        tc_reg_running.loc[index,p,i] = t_reg_running.values + p_reg_running.values
                        tc_reg_running_t.loc[index,p,i] = t_reg_running.values
                        tc_reg_running_p.loc[index,p,i] = p_reg_running.values


        tc_ds = xr.Dataset({"teleconnections_corr":(["index","period","iso"],tc_corr),
                            "teleconnections_temp_corr":(["index","period","iso"],tc_corr_t),
                            "teleconnections_precip_corr":(["index","period","iso"],tc_corr_p),
                            "teleconnections_reg":(["index","period","iso"],tc_reg),
                            "teleconnections_temp_reg":(["index","period","iso"],tc_reg_t),
                            "teleconnections_precip_reg":(["index","period","iso"],tc_reg_p),
                            "teleconnections_corr_running":(["index","period","iso"],tc_corr_running),
                            "teleconnections_temp_corr_running":(["index","period","iso"],tc_corr_running_t),
                            "teleconnections_precip_corr_running":(["index","period","iso"],tc_corr_running_p),
                            "teleconnections_reg_running":(["index","period","iso"],tc_reg_running),
                            "teleconnections_temp_reg_running":(["index","period","iso"],tc_reg_running_t),
                            "teleconnections_precip_reg_running":(["index","period","iso"],tc_reg_running_p)},
                                coords={"iso":(["iso"],iso),
                                        "period":(["period"],periods),
                                        "index":(["index"],indices)})

        tc_ds.attrs["creation_date"] = str(datetime.datetime.now())
        tc_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
        tc_ds.attrs["variable_description"] = "country-level teleconnections from CMIP6 models for T, P, and their sum"
        tc_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_Teleconnections.py"
        tc_ds.attrs["dims"] = "index x period x iso"

        fname_out = loc_out+m+"_historical-"+e+"_country_teleconnections_"+str(y1_hist)+"-"+str(y2_future)+".nc"
        tc_ds.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)
        
        """
        print("E-index teleconnection change using correlation coefficient:",flush=True)
        tc_corr_e_isomean = tc_corr.sel(index="e").mean(dim="iso")
        tc_reg_e_isomean = tc_reg.sel(index="e").mean(dim="iso")
        tc_corr_e_isomean_hist = tc_corr_e_isomean.sel(period="hist")
        tc_corr_e_isomean_future = tc_corr_e_isomean.sel(period="future")
        tc_reg_e_isomean_hist = tc_reg_e_isomean.sel(period="hist")
        tc_reg_e_isomean_future = tc_reg_e_isomean.sel(period="future")

        print((100*(tc_corr_e_isomean_future - tc_corr_e_isomean_hist)/tc_corr_e_isomean_hist).values)

        print("E-index teleconnection change using regression coefficient:",flush=True)
        print((100*(tc_reg_e_isomean_future - tc_reg_e_isomean_hist)/tc_reg_e_isomean_hist).values)
        """
