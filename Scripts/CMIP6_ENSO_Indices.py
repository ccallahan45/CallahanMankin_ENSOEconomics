# ENSO indices from CMIP6 SST
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from eofs.xarray import Eof
import xesmf as xe
## pyfesom?


# locations
loc_out = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Variability_Economics/Data/ENSO_Indices/CMIP6/"
loc_cmip6 = "/dartfs-hpc/rc/lab/C/CMIG/Data/ClimateModels/CMIP6/"

# function for quadratic detrending
def xarray_quadratic_detrend_tll(data):
    # detrends a three-dimensional
    # (time,lat,lon)
    # xarray dataarray separately at
    # each grid point
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
    lt = data.lat
    ln = data.lon
    timevals = xr.DataArray(np.arange(1,len(tm)+1,1),
                        coords=[tm],
                        dims=["time"])
    timevals = timevals.expand_dims(lat=lt,lon=ln)
    timevals = timevals.transpose("time","lat","lon")

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

    predicted_vals = (int_coef + lin_coef*(timevals) + quad_coef*(timevals**2)).transpose("time","lat","lon")
    detrended_data = data - predicted_vals
    return detrended_data

# Functions for ENSO calculations

def calc_sst_anom_region(sst,y1,y2,latmin,latmax,lonmin,lonmax):
    # don't need year indices since we're just doing it over the entire period
    # assumes time x lat x lon
    time_ind = (sst.time.dt.year>=y1) & (sst.time.dt.year<=y2)
    sst_clm = sst[time_ind,:,:].groupby("time.month").mean(dim="time")
    sst_anom = sst.groupby("time.month") - sst_clm
    sst_anom_region = sst_anom.loc[:,latmin:latmax,lonmin:lonmax]
    return(sst_anom_region)

def calc_nonlinear_enso_indices(sst_anomalies):
    # sst anomalies should be time x lat x lon xarray
    # need to define xarray_quadratic_detrend_tll
    # and have Eof package installed
    # returns E-index, C-index, and alpha
    # (alpha = PC1-PC2 nonlinearity coefficient)

    nmodes = 2
    sst_dt = xarray_quadratic_detrend_tll(sst_anomalies)
    nino3 = sst_dt.loc[:,-5:5,210:270].mean(dim=["lat","lon"])
    eof_solver = Eof(sst_dt)
    eofs = eof_solver.eofs(neofs=nmodes)
    pcs = eof_solver.pcs(pcscaling=1,npcs=nmodes)

    for i in np.arange(0,nmodes,1):
        corrcoef = np.corrcoef(pcs[:,i].values,nino3.values)
        # we want mode 1 to be positively correlated with nino3
        if ((i == 0) & (corrcoef[0][1]<0)):
            scaling = -1
        elif ((i == 0) & (corrcoef[0][1]>=0)):
            scaling = 1
        # and mode 2 to be negatively corelated with nino3
        elif ((i == 1) & (corrcoef[0][1]<0)):
            scaling = 1
        elif ((i == 1) & (corrcoef[0][1]>=0)):
            scaling = -1
        else:
            print("ERROR")
            sys.exit()

        eofs[i,:,:] = eofs[i,:,:].values*scaling
        pcs[:,i] = pcs[:,i].values*scaling

    # calculate quadratic fit and get coefficient
    pc1 = pcs[:,0]
    pc2 = pcs[:,1]
    enso_fit = np.polyfit(pc1,pc2,2)
    alpha = enso_fit[0]
    E = (pc1-pc2)/(np.sqrt(2))
    C = (pc1+pc2)/(np.sqrt(2))

    return([E, C, alpha, nino3])


# years
y1_hist = 1850
y2_hist = 2014
y1_ssp = 2015
y2_ssp = 2099
# we'll calculate anomalies relative to historical

# Boundaries
lat_min = -20
lat_max = 20
lon_min = 140
lon_max = 280
# similar to cai et al nature 2020

# new grid
res = 2
lon_new = np.arange(1,359+res,res)
lat_new = np.arange(-89,89+res,res)

var = "tos_Omon"
experiments = ["ssp245","ssp370"] #["ssp126","ssp245","ssp370","ssp585"]
# cycle through experiments and get historical for each one

# loop through experiments, then for each model,
# calculate ENSO indices and write out
models_hist = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical/"+var+"/")) if (x.endswith(".nc"))])
models_prefix_hist = np.unique(np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in models_hist]))

import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning,message="Mean of empty slice")

for exp in experiments:
    print(exp,flush=True)
    loc_in = loc_cmip6+exp+"/"+var+"/"
    loc_in_hist = loc_cmip6+"historical/"+var+"/"

    ## get models
    models = np.array([x for x in sorted(os.listdir(loc_in)) if (x.endswith(".nc"))])
    models_prefix = np.unique(np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in models]))
    print(models_prefix)

    alpha_num = 0
    for m in models_prefix:
        # only proceed if SSP model has a corresponding historical simulation
        mname = m.split("_")[0]
        mreal = m.split("_")[1]

        if m in models_prefix_hist:
            print("*************",flush=True)
            print(m+" ("+str(list(models_prefix).index(m)+1)+"/"+str(len(models_prefix))+")",flush=True)
            print("*************",flush=True)
            if mname in ["CanESM5","CESM2","CESM2-WACCM","CNRM-ESM2-1","CNRM-CM6-1","ACCESS-ESM1-5","ACCESS-CM2","MIROC-ES2L","IPSL-CM6A-LR","FGOALS-g3","GISS-E2-1-G","GISS-E2-1-H"]:
                continue

            # read data and concat
            tos_ds = xr.open_mfdataset(loc_in+var+"_"+mname+"_"+exp+"_"+mreal+"*.nc",concat_dim="time")
            tos_in_ssp = tos_ds.tos.load()
            tm_ssp = tos_ds.time
            tm_ssp_ind = (tm_ssp.dt.year>=y1_ssp)&(tm_ssp.dt.year<=y2_ssp)
            tos_ssp = tos_in_ssp[tm_ssp_ind,:,:]
            tos_in_hist = xr.open_mfdataset(loc_in_hist+var+"_"+mname+"_historical_"+mreal+"*.nc",concat_dim="time").tos.load()
            tm_hist = tos_in_hist.time
            tm_hist_ind = (tm_hist.dt.year>=y1_hist)&(tm_hist.dt.year<=y2_hist)
            tos_hist = tos_in_hist[tm_hist_ind,:,:]
            #tos = xr.concat([tos_hist,tos_ssp],dim="time")
            #del([tos_in_hist,tos_in_ssp])

            #if mname=="MIROC6":
            #    print(tos_hist)
            #    print(tos_ssp)

            # replace calendar to standardize
            cal_hist = pd.date_range(start=str(y1_hist)+"-01-01",end=str(y2_hist)+"-12-31",freq="MS")
            cal_ssp = pd.date_range(start=str(y1_ssp)+"-01-01",end=str(y2_ssp)+"-12-31",freq="MS")
            tos_hist.coords["time"] = cal_hist
            if len(cal_ssp)!=len(tos_ssp.coords["time"].values):
                # limit out models that don't extend the full length of the data
                print(m+" does not appear to span the full "+str(y1_hist)+"-"+str(y2_ssp)+" time period needed")
            else:

                tos_ssp.coords["time"] = cal_ssp

                #if "AWI-CM-1-1" in m:
                if mname in ["AWI-CM-1-1"]: #,"GISS-E2-1-G","CanESM5","GISS-E2-1-H","MIROC-ES2L","KACE-1-0-G"
                    continue
                    # skipping AWI-CM-1-1 for now because it's a weird unstructured grid
                    # could try and install "pyfesom" package for dealing with
                    # the odd grid -- but seems very annoying to deal with

                    # and skipping other models we already finished
                else: #exp=="ssp126": # "ssp245","ssp370","ssp585"
                    fname_1 = [x for x in sorted(os.listdir(loc_in)) if mname+"_"+exp+"_"+mreal in x][0]
                    print(fname_1,flush=True)
                    #tos = tos.load()
                    #print(tos)

                    if "gn" in fname_1:

                        # rename lat/lon coordinates
                        if (("latitude" in tos_hist.coords)&("longitude" in tos_hist.coords)):
                            tos_hist = tos_hist.rename({"latitude":"lat","longitude":"lon"})
                            tos_ssp = tos_ssp.rename({"latitude":"lat","longitude":"lon"})
                        elif (("nav_lon" in tos_hist.coords)&("nav_lat" in tos_hist.coords)):
                            tos_hist = tos_hist.rename({"nav_lat":"lat","nav_lon":"lon"})
                            tos_ssp = tos_ssp.rename({"nav_lat":"lat","nav_lon":"lon"})
                        if len(tos_hist.coords["lat"].values.shape)<3:

                            # regrid to 2 degree grid using xesmf
                            # https://xesmf.readthedocs.io/en/latest/notebooks/Curvilinear_grid.html
                            # https://xesmf.readthedocs.io/en/latest/notebooks/Rectilinear_grid.html
                            # note that despite some input grids being 0-360 and the xesmf routine using
                            # a -180-180 grid, it's smart and knows how to regrid between the two
                            # so the output data will be on the -180-180 lon grid but the data is correct
                            # (See Calculate_Depth_Profile_Nino3.ipynb)

                            grid_out = xe.util.grid_global(2.0,2.0)
                            #grid_out = xr.Dataset({'lat': (['lat'], grid_out_2d.lat[:,0].values),
                            #                            'lon': (['lon'],grid_out_2d.lon[0,:].values)})

                            # regrid
                            regridder_hist = xe.Regridder(tos_hist,grid_out,"bilinear",ignore_degenerate=True)
                            tos_hist_regrid = regridder_hist(tos_hist)
                            regridder_hist.clean_weight_file()  # clean-up
                            del(tos_hist)

                            regridder_ssp = xe.Regridder(tos_ssp,grid_out,"bilinear",ignore_degenerate=True)
                            tos_ssp_regrid = regridder_ssp(tos_ssp)
                            regridder_ssp.clean_weight_file()  # clean-up
                            del(tos_ssp)

                            # now create entirely new variable with one-dimensional coordinates
                            lat = tos_hist_regrid.lat.values[:,0]
                            lon = (tos_hist_regrid.lon.values[0,:])
                            sst_hist = xr.DataArray(tos_hist_regrid.values,
                                                coords=[tos_hist_regrid.time,lat,lon],
                                                dims=["time","lat","lon"])
                            sst_ssp = xr.DataArray(tos_ssp_regrid.values,
                                                coords=[tos_ssp_regrid.time,lat,lon],
                                                dims=["time","lat","lon"])
                            del([tos_hist_regrid,tos_ssp_regrid])
                            sst = xr.concat([sst_hist,sst_ssp],dim="time")
                            del([sst_hist,sst_ssp])

                            if sst.max()>200:
                                sst = sst - 273.15

                            # convert SST to 360 degree longitude
                            sst.coords["lon"] = lon % 360
                            sst_e = sst[:,:,sst.lon.values>180]
                            sst_w = sst[:,:,sst.lon.values<180]
                            sst_final = xr.concat([sst_w,sst_e],dim="lon")
                            del(sst)

                            # anomalies
                            time = sst_final.time

                            sst_anom_region = calc_sst_anom_region(sst_final,y1_hist,y2_hist,lat_min,lat_max,lon_min,lon_max)

                            # enso indices
                            E, C, alpha, nino3 = calc_nonlinear_enso_indices(sst_anom_region)
                            del([sst_final,sst_anom_region])
                            if (alpha<-0.15):
                                alpha_num = alpha_num + 1

                            # dataset
                            enso_ds = xr.Dataset({"e_index":(["time"],E),
                                                        "c_index":(["time"],C),
                                                        "nino3":(["time"],nino3),
                                                        "alpha":([],alpha)},
                                                    coords={"time":(["time"],time)})

                            enso_ds.attrs["creation_date"] = str(datetime.datetime.now())
                            enso_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
                            enso_ds.attrs["variable_description"] = "e_index, c_index, nino3, and alpha (nonlinearity coefficient)"
                            enso_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_ENSO_Indices.py"
                            enso_ds.attrs["dims"] = "time"
                            enso_ds.attrs["grid"] = "regridded to 2x2 rectilinear grid"
                            enso_ds.attrs["detrending"] = "quadratically detrended over entire period"
                            enso_ds.attrs["anomalies"] = "SST anomalies calculated relative to "+str(y1_hist)+"-"+str(y2_hist)

                            fname_out = loc_out+mname+"_"+mreal+"_historical-"+exp+"_ENSO_indices_"+str(y1_hist)+"-"+str(y2_ssp)+".nc"
                            enso_ds.to_netcdf(fname_out,mode="w")
                            print(fname_out,flush=True)

                            print(str(alpha_num)+" models have alpha < -0.15",flush=True)

                        else:
                            print(tos_hist)
                            grid_out = xe.util.grid_global(2.0,2.0)
                            #grid_out = xr.Dataset({'lat': (['lat'], grid_out_2d.lat[:,0].values),
                            #                            'lon': (['lon'],grid_out_2d.lon[0,:].values)})

                            # regrid
                            regridder_hist = xe.Regridder(tos_hist,grid_out,"bilinear",ignore_degenerate=True)
                            tos_hist_regrid = regridder_hist(tos_hist)
                            regridder_hist.clean_weight_file()  # clean-up
                            del(tos_hist)
                            regridder_ssp = xe.Regridder(tos_ssp,grid_out,"bilinear",ignore_degenerate=True)
                            tos_ssp_regrid = regridder_ssp(tos_ssp)
                            regridder_ssp.clean_weight_file()  # clean-up
                            del(tos_ssp)

                            # now create entirely new variable with one-dimensional coordinates
                            lat = tos_hist_regrid.lat.values[:,0]
                            lon = (tos_hist_regrid.lon.values[0,:])
                            sst_hist = xr.DataArray(tos_hist_regrid.values,
                                                coords=[tos_hist_regrid.time,lat,lon],
                                                dims=["time","lat","lon"])
                            sst_ssp = xr.DataArray(tos_ssp_regrid.values,
                                                coords=[tos_ssp_regrid.time,lat,lon],
                                                dims=["time","lat","lon"])
                            del([tos_hist_regrid,tos_ssp_regrid])
                            sst = xr.concat([sst_hist,sst_ssp],dim="time")
                            del([sst_hist,sst_ssp])

                            print(sst)


                    elif ("gr" in fname_1)|("gr1" in fname_1)|("gr2" in fname_1):

                        if tos_hist.lat[0]>tos_hist.lat[10]:
                            tos_hist_for_interp = tos_hist[:,::-1,:]
                            tos_ssp_for_interp = tos_ssp[:,::-1,:]
                        else:
                            tos_hist_for_interp = tos_hist*1.0
                            tos_ssp_for_interp = tos_ssp*1.0
                        del([tos_hist,tos_ssp])

                        if np.amin(tos_hist_for_interp.lon.values)<0:
                            tos_hist_for_interp.coords["lon"] = tos_hist_for_interp.coords["lon"].values % 360
                            tos_ssp_for_interp.coords["lon"] = tos_ssp_for_interp.coords["lon"].values % 360

                        sst_hist = tos_hist_for_interp.interp(lat=lat_new,lon=lon_new)
                        sst_ssp = tos_ssp_for_interp.interp(lat=lat_new,lon=lon_new)
                        sst = xr.concat([sst_hist,sst_ssp],dim="time")
                        del([sst_hist,sst_ssp])
                        if sst.max()>200:
                            sst = sst - 273.15

                        # anomalies
                        time = sst.time
                        sst_anom_region = calc_sst_anom_region(sst,y1_hist,y2_hist,lat_min,lat_max,lon_min,lon_max)
                        del(sst)

                        # enso indices
                        E, C, alpha, nino3 = calc_nonlinear_enso_indices(sst_anom_region)
                        del(sst_anom_region)
                        if (alpha<-0.15):
                            alpha_num = alpha_num + 1

                        # dataset
                        enso_ds = xr.Dataset({"e_index":(["time"],E),
                                                "c_index":(["time"],C),
                                                "nino3":(["time"],nino3),
                                                "alpha":([],alpha)},
                                                coords={"time":(["time"],time)})

                        enso_ds.attrs["creation_date"] = str(datetime.datetime.now())
                        enso_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
                        enso_ds.attrs["variable_description"] = "e_index, c_index, nino3, and alpha (nonlinearity coefficient)"
                        enso_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_ENSO_Indices.py"
                        enso_ds.attrs["dims"] = "time"
                        enso_ds.attrs["grid"] = "regridded to 2x2 rectilinear grid"
                        enso_ds.attrs["detrending"] = "quadratically detrended over entire period"
                        enso_ds.attrs["anomalies"] = "SST anomalies calculated relative to "+str(y1_hist)+"-"+str(y2_hist)

                        fname_out = loc_out+mname+"_"+mreal+"_historical-"+exp+"_ENSO_indices_"+str(y1_hist)+"-"+str(y2_ssp)+".nc"
                        enso_ds.to_netcdf(fname_out,mode="w")
                        print(fname_out,flush=True)

                        print(str(alpha_num)+" models have alpha < -0.15",flush=True)
