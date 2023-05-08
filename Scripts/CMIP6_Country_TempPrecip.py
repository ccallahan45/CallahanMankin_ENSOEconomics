# Country temperature and precipitation in CMIP6
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

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

# locations
loc_out_t = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/CountryTemp/CMIP6/"
loc_out_p = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/CountryPrecip/CMIP6/"
loc_cmip6 = "/dartfs-hpc/rc/lab/C/CMIG/Data/ClimateModels/CMIP6/"
loc_shp = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/ProcessedCountryShapefile/"
loc_pop = "/dartfs-hpc/rc/lab/C/CMIG/Data/Other/GPW/"

# need pr_day and tas_Amon

# years
y1_hist = 1850
y2_hist = 2014
y1_ssp = 2015
y2_ssp = 2099
y1_hist_clm = 1940
y2_hist_clm = 2019
y1_ssp_clm = 2020
y2_ssp_clm = 2099


## functions for flipping longitude
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

# detrending function
def xarray_quadratic_detrend_it(data):
    # detrends a twp-dimensional
    # (iso,time)
    # xarray dataarray separately at
    # each country
    # easy to do, but slow, with a loop
    # so this is a vectorized
    # way of doing it
    # https://stackoverflow.com/questions/38960903/applying-numpy-polyfit-to-xarray-dataset

    def quadratic_trend_coef(x,y):
        pf = np.polyfit(x, y, 2)
        return xr.DataArray(pf[0])
    def linear_trend_coef(x, y):
        pf = np.polyfit(x, y, 2)
        return xr.DataArray(pf[1])
    def intercepts(x, y):
        pf = np.polyfit(x, y, 2)
        return xr.DataArray(pf[2])

    tm = data.time
    countries = data.iso
    timevals = xr.DataArray(np.arange(1,len(tm)+1,1),
                        coords=[tm],
                        dims=["time"])
    timevals = timevals.expand_dims(iso=countries)
    timevals = timevals.transpose("iso","time")

    quad_coef = xr.apply_ufunc(quadratic_trend_coef,
                            timevals,data,
                            vectorize=True,
                            input_core_dims=[["time"],["time"]])
    lin_coef = xr.apply_ufunc(linear_trend_coef,
                            timevals,data,
                            vectorize=True,
                            input_core_dims=[["time"],["time"]])
    int_coef = xr.apply_ufunc(intercepts,
                            timevals,data,
                            vectorize=True,
                            input_core_dims=[["time"],["time"]])

    predicted_vals = (int_coef + lin_coef*(timevals) + quad_coef*(timevals**2)).transpose("iso","time")
    detrended_data = data - predicted_vals
    return detrended_data


# helper functions for spatial averaging
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


# country average function
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


#experiments = ["ssp370","ssp245"]
experiments = ["ssp126","ssp245","ssp370","ssp585"]
# cycle through experiments and get historical for each one

vars = ["tas_Amon","pr_day"]

# new grid
res = 2
lon_new = np.arange(1,359.0+res,res)
lat_new = np.arange(-89.0,89.0+res,res)

# also need population data interpolated to new grid
population_2000 = (xr.open_dataset(loc_pop+"1Degree/gpw_v4_population_count_1degree_lonflip.nc").data_vars["population"])[0,::-1,:]
lat_pop = population_2000.coords["latitude"]
lon_pop = population_2000.coords["longitude"]
pop = xr.DataArray(population_2000.values,
                  coords=[lat_pop,lon_pop],
                  dims=["lat","lon"])
pop_regrid = pop.interp(lat=lat_new,lon=lon_new)
pop_flip = flip_lon_ll(pop_regrid)


# read shapefile
shp = gp.read_file(loc_shp)
iso_shp = shp.ISO3.values
isonums = {i: k for i, k in enumerate(shp.ISO3)}
isonums_rev = {k: i for i, k in enumerate(shp.ISO3)}
shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]

# hist models
t_models_hist = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical/tas_Amon/")) if (x.endswith(".nc"))])
t_models_prefix_hist = np.unique(np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in t_models_hist]))

p_models_hist = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical/pr_day/")) if (x.endswith(".nc"))])
p_models_prefix_hist = np.unique(np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in p_models_hist]))

# warnings
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in")
warnings.filterwarnings("ignore",category=RuntimeWarning,message="Mean of empty slice")
for e in experiments:

    print(e)

    # we'll just do temp and precip separately
    # and then later, when calculating teleconnections,
    # we'll just do models that have both

    # temp models
    loc_in_t = loc_cmip6+e+"/tas_Amon/"
    loc_in_hist_t = loc_cmip6+"historical/tas_Amon/"
    t_models_exp = np.array([x for x in sorted(os.listdir(loc_in_t)) if (x.endswith(".nc"))])
    t_models_prefix_exp = np.unique(np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in t_models_exp]))

    # loop
    for m in t_models_prefix_exp:
        var = "tas_Amon"
        # only proceed if SSP model has a corresponding historical simulation
        mname = m.split("_")[0]
        mreal = m.split("_")[1]
        #model_hist = mname+"_historical_"+mreal

        if m in t_models_prefix_hist:
            print(var+", "+m+" ("+str(list(t_models_prefix_exp).index(m)+1)+"/"+str(len(t_models_prefix_exp))+")",flush=True)
            #if mname in ["ACCESS-ESM1-5","ACCESS-CM2","CanESM5","CESM2-WACCM","CESM2","CNRM-CM-6-1","EC-Earth3","FGOALS-g3","GFDL-ESM4"]:
            #    continue
            if e=="ssp126":
                continue

            # read tas
            tas_ds = xr.open_mfdataset(loc_in_t+"tas_Amon"+"_"+mname+"_"+e+"_"+mreal+"*.nc",concat_dim="time")
            tas_in_ssp = tas_ds.tas.load()
            tm_ssp = tas_ds.time
            tm_ssp_ind = (tm_ssp.dt.year>=y1_ssp)&(tm_ssp.dt.year<=y2_ssp)
            tas_ssp = tas_in_ssp[tm_ssp_ind]
            tas_in_hist = xr.open_mfdataset(loc_in_hist_t+"tas_Amon_"+mname+"_historical_"+mreal+"*.nc",concat_dim="time").tas.load()
            tm_hist = tas_in_hist.time
            tm_hist_ind = (tm_hist.dt.year>=y1_hist)&(tm_hist.dt.year<=y2_hist)
            tas_hist = tas_in_hist[tm_hist_ind]
            tas = xr.concat([tas_hist,tas_ssp],dim="time")

            # standardize calendar and regrid
            if len(tas.time.values)==3000:
                cal = pd.date_range(start=str(y1_hist)+"-01-01",end=str(y2_ssp)+"-12-31",freq="MS")
                tas.coords["time"] = cal

                if tas.max()>200:
                    tas = tas - 273.15

                if (("latitude" in tas.coords)&("longitude" in tas.coords)):
                    tas = tas.rename({"latitude":"lat","longitude":"lon"})
                tas_interp = tas.interp(lat=lat_new,lon=lon_new)
                tas_flip = flip_lon_tll(tas_interp)
                del([tas,tas_interp,tas_hist,tas_ssp])

                # now population weight average by country
                temp = xr_country_average(tas_flip,y1_hist,y2_ssp,"MS",shapes,iso_shp,isonums,True,pop_flip)

                # standardize (for teleconnections)
                temp_std_hist = calc_anom_iso_time(temp.loc[:,str(y1_hist_clm)+"-01-01":str(y2_hist_clm)+"-12-31"],y1_hist_clm,y2_hist_clm)
                temp_std_ssp = calc_anom_iso_time(temp.loc[:,str(y1_ssp_clm)+"-01-01":str(y2_ssp_clm)+"-12-31"],y1_ssp_clm,y2_ssp_clm)
                temp_std = xr.concat([temp_std_hist,temp_std_ssp],dim="time")
                #temp_std = calc_anom_iso_time(temp,y1_hist,y2_hist)
                #temp_std_dt = xarray_quadratic_detrend_it(temp_std)

                # write out
                t_ds = xr.Dataset({"temp":(["iso","time"],temp.loc[:,temp_std.coords["time"]]),
                                    "temp_std":(["iso","time"],temp_std)},
                                    coords={"time":(["time"],temp_std.coords["time"]),
                                            "iso":(["iso"],temp_std.coords["iso"])})

                t_ds.attrs["creation_date"] = str(datetime.datetime.now())
                t_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
                t_ds.attrs["variable_description"] = "temp and temp_std"
                t_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_Country_TempPrecip.py"
                t_ds.attrs["dims"] = "time"
                t_ds.attrs["grid"] = "regridded to 2x2 grid"
                t_ds.attrs["units"] = "degrees c or s.d."

                fname_out = loc_out_t+m+"_historical-"+e+"_country_temperature_monthly_"+str(y1_hist_clm)+"-"+str(y2_ssp_clm)+".nc"
                t_ds.to_netcdf(fname_out,mode="w")
                print(fname_out,flush=True)
            else:
                print(m+" does not appear to span the full 1850-2099 range needed")
        else:
            print(m+" "+var+" not in historical models")

    # now precip models
    loc_in_p = loc_cmip6+e+"/pr_day/"
    loc_in_hist_p = loc_cmip6+"historical/pr_day/"
    p_models_exp = np.array([x for x in sorted(os.listdir(loc_in_p)) if (x.endswith(".nc"))])
    p_models_prefix_exp = np.unique(np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in p_models_exp]))

    # loop

    for m in p_models_prefix_exp:
        var = "pr_day"
        # only proceed if SSP model has a corresponding historical simulation
        mname = m.split("_")[0]
        mreal = m.split("_")[1]

        if m in p_models_prefix_hist:
            print(var+", "+m+" ("+str(list(p_models_prefix_exp).index(m)+1)+"/"+str(len(p_models_prefix_exp))+")",flush=True)
            #if mname in ["ACCESS-CM2","CanESM5","CESM2-WACCM","CESM2","CNRM-CM-6-1","EC-Earth3","FGOALS-g3","CAMS-CSM1-0","IPSL-CM6A-LR","GISS-E2-1-G","GISS-E2-1-H"]:
            #    continue

            # read tas
            pr_ds = xr.open_mfdataset(loc_in_p+"pr_day_"+mname+"_"+e+"_"+mreal+"*.nc",concat_dim="time")
            pr_in_ssp = pr_ds.pr
            units = pr_in_ssp.attrs["units"]
            if units=="kg m-2 s-1":
                scaling = 86400.0 ## convert to mm/day
            else:
                print("precip unit error!")
                sys.exit()

            tm_ssp = pr_ds.time
            tm_ssp_ind = (tm_ssp.dt.year>=y1_ssp)&(tm_ssp.dt.year<=y2_ssp)
            pr_ssp = pr_in_ssp[tm_ssp_ind].load()*scaling
            pr_ssp_mon = pr_ssp.resample(time="MS").sum(dim="time")
            del([pr_ssp,pr_in_ssp])

            pr_in_hist = xr.open_mfdataset(loc_in_hist_p+"pr_day_"+mname+"_historical_"+mreal+"*.nc",concat_dim="time").pr
            tm_hist = pr_in_hist.time
            tm_hist_ind = (tm_hist.dt.year>=y1_hist)&(tm_hist.dt.year<=y2_hist)
            pr_hist = pr_in_hist[tm_hist_ind].load()*scaling
            pr_hist_mon = pr_hist.resample(time="MS").sum(dim="time")
            del([pr_hist,pr_in_hist])
            pr = xr.concat([pr_hist_mon,pr_ssp_mon],dim="time")
            del([pr_hist_mon,pr_ssp_mon])

            if (("latitude" in pr.coords)&("longitude" in pr.coords)):
                pr = pr.rename({"latitude":"lat","longitude":"lon"})
            pr_interp = pr.interp(lat=lat_new,lon=lon_new)
            del(pr)
            pr_flip = flip_lon_tll(pr_interp)
            del(pr_interp)

            # standardize calendar
            if len(pr_flip.time.values)==3000:
                cal = pd.date_range(start=str(y1_hist)+"-01-01",end=str(y2_ssp)+"-12-31",freq="MS")
                pr_flip.coords["time"] = cal

                # now population weight average by country
                precip = xr_country_average(pr_flip,y1_hist,y2_ssp,"MS",shapes,iso_shp,isonums,True,pop_flip)
                del(pr_flip)

                # standardize (for teleconnections)
                precip_std_hist = calc_anom_iso_time(precip.loc[:,str(y1_hist_clm)+"-01-01":str(y2_hist_clm)+"-12-31"],y1_hist_clm,y2_hist_clm)
                precip_std_ssp = calc_anom_iso_time(precip.loc[:,str(y1_ssp_clm)+"-01-01":str(y2_ssp_clm)+"-12-31"],y1_ssp_clm,y2_ssp_clm)
                precip_std = xr.concat([precip_std_hist,precip_std_ssp],dim="time")
                #precip_std = calc_anom_iso_time(precip,y1_hist,y2_hist)
                #precip_std_dt = xarray_quadratic_detrend_it(precip_std)

                # write out
                p_ds = xr.Dataset({"precip":(["iso","time"],precip.loc[:,precip_std.coords["time"]]),
                                    "precip_std":(["iso","time"],precip_std)},
                                    coords={"time":(["time"],precip_std.coords["time"]),
                                            "iso":(["iso"],precip_std.coords["iso"])})

                p_ds.attrs["creation_date"] = str(datetime.datetime.now())
                p_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
                p_ds.attrs["variable_description"] = "precip, precip_std, and precip_std_dt (monthly sum)"
                p_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_Country_TempPrecip.py"
                p_ds.attrs["dims"] = "time"
                p_ds.attrs["grid"] = "regridded to 2x2 grid"
                p_ds.attrs["units"] = "total mm or s.d."

                fname_out = loc_out_p+m+"_historical-"+e+"_country_precipitation_monthly_"+str(y1_hist_clm)+"-"+str(y2_ssp_clm)+".nc"
                p_ds.to_netcdf(fname_out,mode="w")
                print(fname_out,flush=True)
            else:
                print(m+" does not appear to span the full 1850-2099 range needed")
        else:
            print(m+" "+var+" not in historical models")
