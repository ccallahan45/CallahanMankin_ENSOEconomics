# Persistent effect of El Niño on global economic growth

Replication code and data for "Persistent effect of El Niño on global economic growth," by Christopher Callahan and Justin Mankin, published in _Science_, May 2023. 

If you have questions or suggestions, contact Chris Callahan at _Christopher.W.Callahan.GR (at) dartmouth (dot) edu_.

### Organization 

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of most the figures and numbers cited in the text. Some of the files for the climate model projections, raw observational data, and damage estimates are quite large, so they are not provided here. Details are below.

- **Scripts/**: Code required to reproduce the findings of our work is included in this folder. Scripts are written primarily in Python, with R used for the empirical regression analysis. The scripts titled `Fig1.ipynb`, `Fig2.ipynb`, etc. contain the final code that produces the main text figures and most of the numbers cited in the text. 

- **Figures/**: Figures are saved here. This is also where figures will be saved if you run the scripts. Minor postprocessing in Adobe Illustrator was used occasionally (e.g., for Figure 2) but generally the figures output by the scripts will be the same as what appears in the paper. The only exception is Fig. S2, which was made entirely in Illustrator. Also note that these scripts do not produce the tables found in the Supplementary Material. The tables were made in LaTeX and code for creating each table is embedded in the regression script (see below).


### Data 

The full data for the economic damage estimates (underlying Figs. 2 and 4) are large, on the order of gigabytes, so they are not hosted in this repo. However, summarized versions of these datasets are available in the **Data/SummaryData/** folder. `Global_Losses_{1983/1998}.xlsx` have data on global losses from the two historical El Niño events we analyze (1982-83 and 1997-98), corresponding to Figure 2B. `Global_Warming_Damages.xlsx` has data on the global losses expected from changes in El Niño amplitude and teleconnections under various emissions scenarios and discount rates, approximately corresponding to Figure 4A. Damages_ByCountry.xlsx` has country-by-country values for both the historical and future damages, corresponding to Figure S11. 

Raw climate datasets not available in this repository are publicly available as follows:

- The **HadISST** sea surface temperature data are available from the Hadley Centre [here](https://www.metoffice.gov.uk/hadobs/hadisst/).

- The **Berkeley Earth** surface temperature data are available [here](https://berkeleyearth.org/data/). Our analysis uses the combined monthly land and ocean dataset at a 1-degree resolution.

- The **GPCC** precipitation data are available [here](https://psl.noaa.gov/data/gridded/data.gpcc.html). Our analysis uses the "v2020" monthly total precipitation data.

- **CMIP6 temperature, precipitation, and SST** data are generally available from the [Earth System Grid Federation](https://esgf-node.llnl.gov/search/cmip6/). Our analysis uses monthly temperature ("tas_Amon"), daily precip ("pr_day"), and monthly SST ("tos_Omon") data from as many models as were available in ~fall 2021 for SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5. Tables S3-S6 in the Supplementary Information show the models we use, along with the number of realizations from each model. 

If you'd like any other data that isn't provided here (e.g., for the supplementary damages figures), please don't hesitate to let me (Chris) know.

### Scripts

Each script performs a specific step of the analysis. The **main analysis** uses a number of core scripts:

- `Process_Country_TempPrecip.ipynb` calculates observed country-level temperature and precipitation.

- `Observed_ENSO_Indices.ipynb` calculates the observed E- and C-index, as well as the Niño3 and Niño3.4 ENSO indices (and plots Fig. S1).

- `Observed_Teleconnections.ipynb` uses the ENSO indices as well as country-level temp and precip to calculate the teleconnection indices used in the paper.

- `Construct_ENSO_Panel.ipynb` assembles the above climate data along with economic data into a panel dataset used in the regression analysis.

- `ENSO_Growth_Regression.R` performs the main regression analysis. Many different forms of the regression are performed (sensitivity analyses, etc.) with bootstraps for each one, so this script can take hours to run in full.

- `ENSO_Event_Damages.py` calculates damages that result from individual historical El Niño events. 

- `CMIP6_ENSO_Indices.py`, `CMIP6_Country_TempPrecip.py`, and `CMIP6_Teleconnections.py` calculate climate model-based versions of the ENSO indices and teleconnections over the 20th and 21st centuries. These scripts take hours to run, even when on a high-performance computing cluster, and the input data is generally not provided (see above) due to large file sizes. The output datasets from these scripts are provided in the **Data** folder, so the rest of the analysis will still run. 

- `ENSO_Future_Damages.py` calculates future economic losses from warming-driven changes in ENSO. This script takes several hours to run on an HPC cluster. 

- `Fig1.ipynb`, `Fig2.ipynb`, `Fig3.ipynb`, and `Fig4.ipynb` read the intermediate data, perform some final calculations, and plot the main figures.

There are also several **supplementary analyses** that use additional scripts:

- `Plot_Regression_Sensitivity.ipynb` plots many of the sensitivity analyses and alternative tests found in the Supplementary Material (Figs. S3, S4, S5, S8, S9, S14) based on the calculations from `ENSO_Growth_Regression.R`.

- `DistributedLag_Testing.R` performs the synthetic data simulations used to evaluate models with different numbers of lags.

- `Plot_DID_Testing.ipynb` evaluates and plots the alternative regression specifications that use a combination of country and year fixed effects, as well as the tests of effect heterogeneity. This script plots Figs. S6 and S7. 

- `Plot_Marginal_Effects.ipynb` evaluates and compares the marginal effects of the E-index and C-index, and plots Fig. S10 accordingly.

- `Plot_CountryLevel_Losses.ipynb` evaluates country-level economic damages (rather than just global damages) from the historical El Niño events and future ENSO changes, and plots Fig. S11 accordingly.

- `Plot_Damage_Sensitivity.ipynb` plots the results of our future damages calculations when making alternative analytical choices such as using only one realization per climate model (Fig. S15).

- `Plot_Growth_Trends.ipynb` plots the histogram of linear trends in country-level economic growth (Fig. S15).

- `Plot_Persistence_Schematic.ipynb` plots the illustration of partially persistent growth effects (Fig. S16). 

- `Calculate_Gridded_Teleconnections.py` calculates grid-cell teleconnections (rather than country-level) from the observational data. This script takes several hours on an HPC cluster. `Plot_Teleconnection_Heterogeneity.ipynb` then plots these grid-cell-level teleconnections alongside teleconnections calculated in moving windows (from `Observed_Teleconnections.ipynb`) in Fig. S17.

