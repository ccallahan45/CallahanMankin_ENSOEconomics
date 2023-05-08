# Teleconnections at the grid cell level
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

## Mechanics
# dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from rasterio import features
from affine import Affine
import geopandas as gp
import descartes
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from scipy import signal, stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols as reg

# Data locations

loc_shp = "../Data/ProcessedCountryShapefile/"
loc_panel = "../Data/Panel/"
loc_enso = "../Data/ENSO_Indices/"
loc_country_temp = "../Data/CountryTemp/"
loc_country_precip = "../Data/CountryPrecip/"
loc_temp_best = "/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/BerkeleyEarth/"
loc_precip = "/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/GPCC/"
loc_pop = "../Data/GPW/"
loc_out = "../Data/Teleconnections/"
loc_out_temp = "../Data/CountryTemp/"
loc_out_precip = "../Data/CountryPrecip/"

# shapefile averaging functions
def transform_from_latlon(lat, lon):
    # Written by Alex Gottlieb
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale
def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))

def flip_lon_tll(da):
    # flip 360 to 180 lon
    # for time-lat-lon xarray dataarray

    # get coords
    lat_da = da.coords["lat"]
    lon_da = da.coords["lon"]
    time_da = da.coords["time"]

    # flip lon
    lon_180 = (lon_da.values + 180) % 360 - 180

    # new data array
    da_180 = xr.DataArray(da.values,
                          coords=[time_da,lat_da.values,lon_180],
                          dims=["time","lat","lon"])

    # flip dataarray so it goes from -180 to 180
    # (instead of 0-180, -180-0)
    lon_min_neg = np.amin(lon_180[lon_180<0])
    lon_max_neg = np.amax(lon_180[lon_180<0])
    lon_min_pos = np.amin(lon_180[lon_180>=0])
    lon_max_pos = np.amax(lon_180[lon_180>=0])
    da_180_flip = xr.concat([da_180.loc[:,:,lon_min_neg:lon_max_neg],
                             da_180.loc[:,:,lon_min_pos:lon_max_pos]],
                            dim="lon")
    return(da_180_flip)

def flip_lon_ll(da):
    # flip 360 to 180 lon
    # for lat-lon xarray dataarray

    # get coords
    lat_da = da.coords["lat"]
    lon_da = da.coords["lon"]

    # flip lon
    lon_180 = (lon_da.values + 180) % 360 - 180

    # new data array
    da_180 = xr.DataArray(da.values,
                          coords=[lat_da,lon_180],
                          dims=["lat","lon"])

    # flip dataarray so it goes from -180 to 180
    # (instead of 0-180, -180-0)
    lon_min_neg = np.amin(lon_180[lon_180<0])
    lon_max_neg = np.amax(lon_180[lon_180<0])
    lon_min_pos = np.amin(lon_180[lon_180>=0])
    lon_max_pos = np.amax(lon_180[lon_180>=0])
    da_180_flip = xr.concat([da_180.loc[:,lon_min_neg:lon_max_neg],
                             da_180.loc[:,lon_min_pos:lon_max_pos]],
                            dim="lon")
    return(da_180_flip)


def xr_country_average(data,y1,y2,freq,shapes,iso_list,isonums_dict,weight=False,weightdata=None):
    time = pd.date_range(start=str(y1)+"-01-01",end=str(y2)+"-12-31",freq=freq)
    country_array = xr.DataArray(np.full((len(iso_list),len(time)),np.nan),
                                 coords=[iso_list,time],
                                 dims=["iso","time"])

    isoraster = rasterize(shapes,data.drop("time").coords)
    data.coords["country"] = isoraster
    if weight:
        weightdata.coords["country"] = isoraster
        countrymean = ((data * weightdata).groupby("country").sum())/(weightdata.groupby("country").sum())
    else:
        countrymean = data.groupby("country").mean()

    isocoord = np.array([isonums_dict[n] for n in countrymean.coords["country"].values])
    country_array.loc[isocoord,:] = countrymean.transpose("country","time").values
    return(country_array)

def xarray_linear_detrend(data):
    # detrends a three-dimensional
    # (time,lat,lon)
    # xarray dataarray separately at
    # each grid point
    # easy to do, but slow, with a loop
    # so this is a vectorized
    # way of doing it
    # https://stackoverflow.com/questions/38960903/applying-numpy-polyfit-to-xarray-dataset

    def linear_trend(x, y):
        pf = np.polyfit(x, y, 1)
        return xr.DataArray(pf[0])
    def intercepts(x, y):
        pf = np.polyfit(x, y, 1)
        return xr.DataArray(pf[1])

    tm = data.time
    lt = data.lat
    ln = data.lon
    timevals = xr.DataArray(np.arange(1,len(tm)+1,1),
                        coords=[tm],
                        dims=["time"])
    timevals = timevals.expand_dims(lat=lt,lon=ln)
    timevals = timevals.transpose("time","lat","lon")

    trends = xr.apply_ufunc(linear_trend,
                            timevals,data,
                            vectorize=True,
                            input_core_dims=[["time"],["time"]])
    intcpts = xr.apply_ufunc(intercepts,
                             timevals,data,
                             vectorize=True,
                             input_core_dims=[["time"],["time"]])

    predicted_vals = (intcpts + trends*timevals).transpose("time","lat","lon")
    detrended_data = data - predicted_vals
    return detrended_data

### analysis

y1_final = 1960
y2_final = 2019

import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in .resample")

## population data

# half degree
pop_halfdegree = (xr.open_dataset(loc_pop+"30ArcMinute/gpw_v4_une_atotpopbt_cntm_30_min_lonflip.nc").data_vars["population"])[0,::-1,:]
pop_flip_halfdegree = flip_lon_ll(pop_halfdegree.rename({"latitude":"lat","longitude":"lon"}))
lat_min_halfdegree = np.amin(pop_flip_halfdegree.lat.values)
lat_max_halfdegree = np.amax(pop_flip_halfdegree.lat.values)

# one degree
pop_onedegree = (xr.open_dataset(loc_pop+"1Degree/gpw_v4_population_count_1degree_lonflip.nc").data_vars["population"])[0,::-1,:]
pop_flip_onedegree = flip_lon_ll(pop_onedegree.rename({"latitude":"lat","longitude":"lon"}))
lat_min_onedegree = np.amin(pop_flip_onedegree.lat.values)
lat_max_onedegree = np.amax(pop_flip_onedegree.lat.values)

## shapefile

shp = gp.read_file(loc_shp)
iso_shp = shp.ISO3.values
isonums = {i: k for i, k in enumerate(shp.ISO3)}
isonums_rev = {k: i for i, k in enumerate(shp.ISO3)}
shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]

## gridded temp from berkeley earth

y1_temp = 1850
y2_temp = 2022

tmean_best = xr.open_dataset(loc_temp_best+"BerkeleyEarth_Land_and_Ocean_LatLong1.nc").rename({"latitude":"lat","longitude":"lon"})
tmean_anom = tmean_best.temperature
tmean_clm = tmean_best.climatology
lsm1 = tmean_best.land_mask
lsm = lsm1.where(lsm1==1,np.nan)

tmean_time_new = pd.date_range(start=str(y1_temp)+"-01-01",end=str(y2_temp)+"-12-31",freq="MS")
tmean_time_round = np.floor(tmean_anom.time.values)
tmean_anom_yrs = tmean_anom[(tmean_time_round>=y1_temp)&(tmean_time_round<(y2_temp+1))]
tmean_anom_yrs.coords["time"] = tmean_time_new

tmean_be = xr.DataArray(np.full(tmean_anom_yrs.shape,np.nan),
                     coords=[tmean_time_new,tmean_anom_yrs.coords["lat"],tmean_anom_yrs.coords["lon"]],
                     dims=["time","lat","lon"])
for mm in np.arange(0,12,1):
    print(mm)
    indices = tmean_time_new.month==(mm+1)
    tmean_be[indices,:,:] = tmean_anom_yrs[indices,:,:] + tmean_clm[mm,:,:]
    
tmean = tmean_be.loc[str(y1_final)+"-01-01":str(y2_final)+"-12-31",:,:] #.loc[:,lat_min_onedegree:lat_max_onedegree,:] #*lsm

## precip

precip_in = xr.open_dataset(loc_precip+"precip.mon.total.v2020.nc").precip

# regrid
precip = precip_in[:,::-1,:].loc[str(y1_final)+"-01-01":str(y2_final)+"-12-31",:,:]

tmean.coords["lon"] = tmean.lon.values % 360
tmean_final = xr.concat([tmean.loc[:,:,0.5:179.5],tmean.loc[:,:,180.5:359.5]],dim="lon")
del([tmean,tmean_be,tmean_anom])

precip_final = precip.interp(lat=tmean_final.lat,lon=tmean_final.lon)
del([precip,precip_in])

## ENSO indices

y1_enso = 1960
y2_enso = 2019
enso_in = xr.open_dataset(loc_enso+"obs_ENSO_indices_monthly_"+str(y1_enso)+"-"+str(y2_enso)+".nc")
e_monthly = enso_in.e_index
c_monthly = enso_in.c_index

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

Eshift = e_monthly.shift(time=1)
Eshift.coords["time"] = pd.date_range(start=str(y1_enso)+"-01-01",end=str(y2_enso)+"-12-31",freq="MS")
#E_yr = Eshift[Eshift.time.dt.month<=3].resample(time="YS").mean(dim="time")
Cshift = c_monthly.shift(time=1)
Cshift.coords["time"] = pd.date_range(start=str(y1_enso)+"-01-01",end=str(y2_enso)+"-12-31",freq="MS")
#C_yr = Cshift[Cshift.time.dt.month<=3].resample(time="YS").mean(dim="time")
E_yr = monthly_to_yearly_mean(Eshift[Eshift.time.dt.month<=3])
C_yr = monthly_to_yearly_mean(Cshift[Cshift.time.dt.month<=3])
E_yr.coords["time"] = E_yr.time.dt.year.values
C_yr.coords["time"] = C_yr.time.dt.year.values

## calculate teleconnections

tmean_abs_tc = tmean_final[(tmean_final.time.dt.year>=y1_enso)&(tmean_final.time.dt.year<=y2_enso),:,:]
precip_abs_tc = precip_final[(precip_final.time.dt.year>=y1_enso)&(precip_final.time.dt.year<=y2_enso),:,:]

## calc anoms
tmean_mean = tmean_abs_tc.groupby("time.month").mean("time")
tmean_std = tmean_abs_tc.groupby("time.month").std("time")
tmean_tc = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    tmean_abs_tc.groupby("time.month"),
    tmean_mean,tmean_std)
precip_mean = precip_abs_tc.groupby("time.month").mean("time")
precip_std = precip_abs_tc.groupby("time.month").std("time")
precip_tc = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    precip_abs_tc.groupby("time.month"),
    precip_mean,precip_std)




mons = [-6,-7,-8,-9,-10,-11,-12,1,2,3,4,5,6,7,8]
# (-) means year t, (+) means year t+1
inds = ["e","c"]

def linear_trend(x, y):
    pf = np.polyfit(x, y, 1)
    return xr.DataArray(pf[0])
def reg_temp(t,p):
    df = pd.DataFrame({"t":t,"p":p,"enso":enso.values})
    param = reg("t ~ enso + p",data=df).fit().params["enso"]
    return(param)
def reg_precip(t,p):
    df = pd.DataFrame({"t":t,"p":p,"enso":enso.values})
    param = reg("p ~ enso + t",data=df).fit().params["enso"]
    return(param)
def partial_correlation(x,y,z):
    from statsmodels.formula.api import ols as reg
    df = pd.DataFrame({"x":x,"y":y,"z":z})
    x_z_resids = reg("x~z",data=df).fit().resid.values
    y_z_resids = reg("y~z",data=df).fit().resid.values
    corr, p = stats.pearsonr(x_z_resids,y_z_resids)
    return(corr)

lat = tmean_tc.lat.values
lon = tmean_tc.lon.values

t_coefs = xr.DataArray(np.full((len(inds),len(lat),len(lon)),np.nan),
                        coords=[inds,lat,lon],dims=["index","lat","lon"])
p_coefs = xr.DataArray(np.full((len(inds),len(lat),len(lon)),np.nan),
                        coords=[inds,lat,lon],dims=["index","lat","lon"])
p_coefs2 = xr.DataArray(np.full((len(inds),len(lat),len(lon)),np.nan),
                        coords=[inds,lat,lon],dims=["index","lat","lon"])

for ind in inds:
    if ind=="e":
        enso = E_yr[1:]
    elif ind=="c":
        enso = C_yr[1:]
    print(ind,flush=True)

    t_coefs_m = xr.DataArray(np.full((len(mons),len(lat),len(lon)),np.nan),
                            coords=[mons,lat,lon],dims=["month","lat","lon"])
    p_coefs_m = xr.DataArray(np.full((len(mons),len(lat),len(lon)),np.nan),
                            coords=[mons,lat,lon],dims=["month","lat","lon"])

    for mm in np.arange(0,len(mons),1):
        m = mons[mm]
        print(m,flush=True)

        if m<0:
            tmean_m = xarray_linear_detrend(tmean_tc[tmean_tc.time.dt.month==np.abs(m),:,:].shift(time=1)[1:])
            precip_m = xarray_linear_detrend(precip_tc[precip_tc.time.dt.month==np.abs(m),:,:].shift(time=1)[1:])
        else:
            tmean_m = xarray_linear_detrend(tmean_tc[tmean_tc.time.dt.month==m,:,:][1:])
            precip_m = xarray_linear_detrend(precip_tc[precip_tc.time.dt.month==m,:,:][1:])

        # check time elapsed
        import time
        start = time.time()

        for j in np.arange(0,len(lat),1):
            for k in np.arange(0,len(lon),1):
                #print(lat[j])
                #print(lon[k])
                tmean_jk = tmean_m[:,j,k]
                precip_jk = precip_m[:,j,k]

                if (~np.any(np.isnan(tmean_jk)))&(~np.any(np.isnan(precip_jk))):
                    #t_coefs_m[mm,j,k] = reg_temp(tmean_jk,precip_jk)
                    #p_coefs_m[mm,j,k] = reg_precip(tmean_jk,precip_jk)
                    t_coefs_m[mm,j,k] = partial_correlation(enso,tmean_jk,precip_jk)
                    p_coefs_m[mm,j,k] = partial_correlation(enso,precip_jk,tmean_jk)

        end = time.time()
        print((end-start)/60.,flush=True)

    t_coefs.loc[ind,:,:] = np.abs(t_coefs_m).rolling(month=3).mean().max(dim="month")
    p_coefs.loc[ind,:,:] = np.abs(p_coefs_m).rolling(month=3).mean().max(dim="month")
    #t_regcoefs.loc[ind,:,:] = np.abs(t_regcoefs_m).max(dim="month")
    #p_regcoefs.loc[ind,:,:] = np.abs(p_regcoefs_m).max(dim="month")
    p_max = p_coefs_m.rolling(month=3).mean().max(dim="month")
    p_min = p_coefs_m.rolling(month=3).mean().min(dim="month")
    for j in np.arange(0,len(lat),1):
        for k in np.arange(0,len(lon),1):
            if np.abs(p_max[j,k].values)>np.abs(p_min[j,k].values):
                p_coefs2.loc[ind,lat[j],lon[k]] = p_max[j,k].values
            elif np.abs(p_min[j,k].values)>np.abs(p_max[j,k].values):
                p_coefs2.loc[ind,lat[j],lon[k]] = p_min[j,k].values

desc = "t_corr_e: max monthly e-t partial correlation; "\
        "p_corr_e: max monthly e-p partial correlation; "\
        "p_corr2_e: max monthly e-p partial correlation, sign preserved;"\
        "t_corr_c: max monthly c-t partial correlation; "\
        "p_corr_c: max monthly c-p partial correlation; "\
        "p_corr2_c: max monthly c-p partial correlation, sign preserved;"\
        "t_p_corr_e: sum of correlation coefficients, e-index;"\
        "t_p_corr_c: sum of correlation coefficients, c-index"

ds = xr.Dataset({"t_corr_e":(["lat","lon"],t_coefs.loc["e",:,:]),
                "p_corr_e":(["lat","lon"],p_coefs.loc["e",:,:]),
                "p_corr2_e":(["lat","lon"],p_coefs2.loc["e",:,:]),
                "t_corr_c":(["lat","lon"],t_coefs.loc["c",:,:]),
                "p_corr_c":(["lat","lon"],p_coefs.loc["c",:,:]),
                "p_corr2_c":(["lat","lon"],p_coefs2.loc["c",:,:]),
                "t_p_corr_e":(["lat","lon"],t_coefs.loc["e",:,:]+p_coefs.loc["e",:,:]),
                "t_p_corr_c":(["lat","lon"],t_coefs.loc["c",:,:]+p_coefs.loc["c",:,:])},
               coords={"lat":(["lat"],lat),
                      "lon":(["lon"],lon)})

ds.attrs["creation_date"] = str(datetime.datetime.now())
ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
ds.attrs["variable_description"] = desc
ds.attrs["created_from"] = os.getcwd()+"/Calculate_Gridded_Teleconnections.py"

fname_out = loc_out+"ENSO_gridded_teleconnections_DJF_"+str(y1_enso)+"-"+str(y2_enso)+".nc"
ds.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
