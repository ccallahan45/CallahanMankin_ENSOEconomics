# Historical economic damages from El Nino events
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Dependencies
import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from scipy import stats, signal

# Years
y1 = 1960
y2 = 2019

# Data locations
loc_coefs = "../Data/RegressionResults/"
loc_panel = "../Data/Panel/"
loc_enso = "../Data/ENSO_Indices/"
loc_out = "../Data/Historical_Damages/"

# Read panel
print("reading initial data and regression coefficients",flush=True)
panel = pd.read_csv(loc_panel+"ENSO_Growth_Panel_"+str(y1)+"-"+str(y2)+".csv",index_col=0)
E = panel.loc[(panel.iso=="USA"),"e"].values
C = panel.loc[(panel.iso=="USA"),"c"].values

# Monthly E-index
enso_in = xr.open_dataset(loc_enso+"obs_ENSO_indices_monthly_"+str(y1)+"-"+str(y2)+".nc")
e_monthly = enso_in.e_index
c_monthly = enso_in.c_index

# Now read regression coefficients
# We'll use the ones that exclude 1983 and 1998
# So it's an out of sample test that doesn't have the bias
# of the coinciding financial crises

phase = "ENSO"
#enso_var = "e_e-and-c"
nlag = 5
trend = "none"
response_var = "gr_pwt_frac"
enso_coefs_e = pd.read_csv(loc_coefs+phase+"_teleconnection-interacted_coefs_bootstrap-dist_lag"+str(nlag)+"_e_e-and-c_"+response_var+"_trend"+trend+"_sensitivity_noextreme.csv",index_col=0)
int_coefs_e = pd.read_csv(loc_coefs+phase+"_teleconnection-interaction_coefs_bootstrap-dist_lag"+str(nlag)+"_e_e-and-c_"+response_var+"_trend"+trend+"_sensitivity_noextreme.csv",index_col=0)
enso_coefs_c = pd.read_csv(loc_coefs+phase+"_teleconnection-interacted_coefs_bootstrap-dist_lag"+str(nlag)+"_c_e-and-c_"+response_var+"_trend"+trend+"_sensitivity_noextreme.csv",index_col=0)
int_coefs_c = pd.read_csv(loc_coefs+phase+"_teleconnection-interaction_coefs_bootstrap-dist_lag"+str(nlag)+"_c_e-and-c_"+response_var+"_trend"+trend+"_sensitivity_noextreme.csv",index_col=0)
boot = enso_coefs_e.boot.values
nboot = len(boot)

lags = np.arange(0,nlag+1,1)
coefs_enso_xr_e = xr.DataArray(enso_coefs_e.iloc[:,1:].values,
                            coords=[boot,lags],dims=["boot","lag"])
coefs_int_xr_e = xr.DataArray(int_coefs_e.iloc[:,1:].values,
                            coords=[boot,lags],dims=["boot","lag"])
coefs_enso_xr_c = xr.DataArray(enso_coefs_c.iloc[:,1:].values,
                            coords=[boot,lags],dims=["boot","lag"])
coefs_int_xr_c = xr.DataArray(int_coefs_c.iloc[:,1:].values,
                            coords=[boot,lags],dims=["boot","lag"])


# Now get actual GDPpc, growth, and teleconnections
print("assembling observed country-year data",flush=True)

iso = np.unique(panel.iso.values)
time = np.arange(y1,y2+1,1)
actual_gpc = xr.DataArray(np.full((len(iso),len(time)),np.nan),
                     coords=[iso,time],
                     dims=["iso","time"])
tau_e = xr.DataArray(np.full((len(iso),len(time)),np.nan),
                     coords=[iso,time],
                     dims=["iso","time"])
tau_c = xr.DataArray(np.full((len(iso),len(time)),np.nan),
                     coords=[iso,time],
                     dims=["iso","time"])
actual_pop = xr.DataArray(np.full((len(iso),len(time)),np.nan),
                     coords=[iso,time],
                     dims=["iso","time"])

tc_metric = "t_p_corr_running"
for jj in np.arange(0,len(iso),1):
    actual_gpc[jj,:] = panel.loc[panel["iso"]==iso[jj],"gpc_pwt"].values
    tau_e[jj,:] = panel.loc[panel["iso"]==iso[jj],tc_metric+"_e"].values
    tau_c[jj,:] = panel.loc[panel["iso"]==iso[jj],tc_metric+"_c"].values
    population = panel.loc[panel["iso"]==iso[jj],"pop_pwt"].values
    actual_pop[jj,:] = population

#print(tau_e.sel(iso="PER"))

actual_growth1 = actual_gpc.diff(dim="time",n=1)
actual_frac_growth = actual_growth1/(actual_gpc.loc[:,:(y2-1)].values)
growth_nans = xr.DataArray(np.full((len(iso),1),np.nan),
                           coords=[iso,[y1]],
                           dims=["iso","time"])
actual_growth = xr.concat([growth_nans,actual_frac_growth],dim="time")

# Use this data to calculate marginal effects for each country
print("calculating marginal effects",flush=True)

me_nonlimited_e = coefs_enso_xr_e + coefs_int_xr_e*tau_e
# potentially set marginal effects to zero for countries whose
# confidence intervals cross zero
me_lower_e = me_nonlimited_e.sum(dim="lag").quantile(0.025,dim="boot")
me_upper_e = me_nonlimited_e.sum(dim="lag").quantile(0.975,dim="boot")
me_e = me_nonlimited_e.where((me_lower_e<=0)&(me_upper_e<=0),0.0)
#me = me_nonlimited*1.0
# me : boot x lag x iso x time

me_nonlimited_c = coefs_enso_xr_c + coefs_int_xr_c*tau_c
me_lower_c = me_nonlimited_c.sum(dim="lag").quantile(0.025,dim="boot")
me_upper_c = me_nonlimited_c.sum(dim="lag").quantile(0.975,dim="boot")
me_c = me_nonlimited_c.where((me_lower_c<=0)&(me_upper_c<=0),0.0)


# Calculate damages from specific El Nino events
print("calculating damages from extreme el nino events",flush=True)

events = [1983,1998]
time_final = np.arange(events[0]-1,y2+1,1) # start damages timeseries at the year
                                            # before the first event, not 1960
actual_e = xr.DataArray(E[time>=(events[0]-1)],
                        coords=[time_final],
                        dims=["time"]).expand_dims(iso=actual_gpc.iso)
actual_c = xr.DataArray(C[time>=(events[0]-1)],
                        coords=[time_final],
                        dims=["time"]).expand_dims(iso=actual_gpc.iso)

# useful functions
def create_growth_arrays(dg,iso,time,boot):
    cds = [boot,time,iso]
    dms = ["boot","time","iso"]
    cf_gdp = xr.DataArray(np.full(dg.values.shape,np.nan),
                         coords=cds,dims=dms)
    cf_growth = xr.DataArray(np.full(dg.values.shape,np.nan),
                         coords=cds,dims=dms)
    return([cf_gdp,cf_growth])

def calc_counterfactual_gdp(orig_growth,orig_gpc,delta_growth,iso,time,boot):
    yrs = time
    cf_gpc, cf_growth = create_growth_arrays(delta_growth,iso,time,boot)
    cf_gr = orig_growth.loc[:,yrs].transpose("time","iso") + delta_growth
    cf_growth[:,:,:] = cf_gr.transpose("boot","time","iso").values
    for yy in np.arange(1,len(yrs),1):
        if yy == 1:
            cf_gdp_year = orig_gpc.loc[:,yrs[yy-1]]
        else:
            cf_gdp_year = cf_gpc.loc[:,yrs[yy-1],:]
        cf_gpc.loc[:,yrs[yy],:] = (cf_gdp_year+(cf_growth.loc[:,yrs[yy],:]*cf_gdp_year)).transpose("boot","iso")
    return([cf_gpc, cf_growth])
    #return(cf_gpc)

## loop twice -- once with and once without benefits of La Nina following event
for k in [0,1]:
    if k == 0:
        print("No La Nina benefits")
    elif k == 1:
        print("Including La Nina benefits")

    for e in np.arange(0,len(events),1):
        print(events[e],flush=True)

        # create counterfactual ENSO
        cf_e = xr.DataArray(actual_e.values,coords=[iso,time_final],dims=["iso","time"])
        time_expand = xr.DataArray(time_final,coords=[time_final],dims=["time"]).expand_dims(iso=actual_gpc.iso)
        cf_e = cf_e.where(time_expand!=events[e],0.0)

        ## incorporate La Nina events in the first and second year following the La Nina event
        if k == 0: # no La Nina benefits
            cf_c = xr.DataArray(actual_c.values,coords=[iso,time_final],dims=["iso","time"])
        else: # yes La Nina benefits
            cf_c = xr.DataArray(actual_c.values,coords=[iso,time_final],dims=["iso","time"])
            cf_c = cf_c.where((time_expand<=events[e])|(time_expand>events[e]+2),0.0)

        # calculate change in growth
        # incorporate both counterfactual E and C indices
        me_e_initial =  me_e.loc[:,0,:,time_final]
        me_c_initial = me_c.loc[:,0,:,time_final]
        delta_growth_events = ((me_e_initial*cf_e)+(me_c_initial*cf_c)) - ((me_e_initial*actual_e)+(me_c_initial*actual_c))
        for l in np.arange(1,nlag+1,1):
            print(l,flush=True)
            me_e_l = me_e.loc[:,l,:,:]
            me_c_l = me_c.loc[:,l,:,:]
            delta = ((me_e_l*cf_e.shift(time=l))+(me_c_l*cf_c.shift(time=l))) - ((me_e_l*actual_e.shift(time=l))+(me_c_l*actual_c.shift(time=l)))
            delta_growth_events = delta_growth_events + delta.where(~np.isnan(delta),0.0)

        # now counterfactual GPC and gfrowth
        cf_gpc_event, cf_gr_event = calc_counterfactual_gdp(actual_growth.loc[:,time_final],actual_gpc.loc[:,time_final],
                                     delta_growth_events.transpose("boot","time","iso"),
                                     iso,time_final,boot)
        if e == 0:
            cf_gpc_events = cf_gpc_event.expand_dims("event")
            cf_gr_events = cf_gr_event.expand_dims("event")
        else:
            cf_gpc_events = xr.concat([cf_gpc_events,cf_gpc_event],dim="event")
            cf_gr_events = xr.concat([cf_gr_events,cf_gr_event],dim="event")

    cf_gpc_events.coords["event"] = events
    cf_gr_events.coords["event"] = events

    if k == 0:
        cf_gpc_events_noln = cf_gpc_events
        cf_gr_events_noln = cf_gr_events
    elif k == 1:
        cf_gpc_events_ln = cf_gpc_events
        cf_gr_events_ln = cf_gr_events

actual_gdp = actual_gpc*actual_pop
cf_gdp_events_noln = cf_gpc_events_noln*actual_pop.loc[:,time_final]
cf_gdp_events_ln = cf_gpc_events_ln*actual_pop.loc[:,time_final]

# calculate change in GDP
gpc_change_events_pct = 100*(actual_gpc.loc[:,time_final] - cf_gpc_events_ln)/cf_gpc_events_ln

# calculate changes in global income
gdp_diff_events_noln = actual_gdp.loc[:,time_final] - cf_gdp_events_noln
global_gdp_diff_events_noln = gdp_diff_events_noln.sum(dim="iso")

gdp_diff_events_ln = actual_gdp.loc[:,time_final] - cf_gdp_events_ln
global_gdp_diff_events_ln = gdp_diff_events_ln.sum(dim="iso")


# write out
events_damage_ds = xr.Dataset({"gpc_change_pct":(["event","iso","time","boot"],gpc_change_events_pct.transpose("event","iso","time","boot")),
                                "gdp_change_noln":(["event","iso","time","boot"],gdp_diff_events_noln.transpose("event","iso","time","boot")),
                                "gdp_change_ln":(["event","iso","time","boot"],gdp_diff_events_ln.transpose("event","iso","time","boot")),
                                "cf_gpc":(["event","iso","time","boot"],cf_gpc_events_ln.transpose("event","iso","time","boot")),
                                "cf_gr":(["event","iso","time","boot"],cf_gr_events_ln.transpose("event","iso","time","boot")),
                                "actual_gpc":(["iso","time"],actual_gpc.loc[:,time_final])},
                            coords={"iso":(["iso"],iso),
                                    "time":(["time"],time_final),
                                    "event":(["event"],events),
                                    "boot":(["boot"],boot)})

events_damage_ds.attrs["creation_date"] = str(datetime.datetime.now())
events_damage_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
events_damage_ds.attrs["variable_description"] = "country-level economic damages/changes from extreme el nino events relative to no-event counterfactual"
events_damage_ds.attrs["created_from"] = os.getcwd()+"/ENSO_Event_Damages.py"
events_damage_ds.attrs["dims"] = "iso, time"

fname_out = loc_out+"ENSO_damages_historical_extreme_elnino_e-and-c_"+str(time_final[0])+"-"+str(y2)+".nc"
events_damage_ds.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
