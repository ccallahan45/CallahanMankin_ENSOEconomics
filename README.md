# Persistent effect of El Niño on global economic growth

Replication code and data for "Persistent effect of El Ni\~{n}o on global economic growth," by Christopher Callahan and Justin Mankin, published in _Science_, May 2023. 

If you have questions or suggestions, contact Chris Callahan at _Christopher.W.Callahan.GR (at) dartmouth (dot) edu_.

### Organization 

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of most the figures and numbers cited in the text. Some of the files for the climate model projections and raw observational data are quite large, so they are not provided here. Details are below.

- **Scripts/**: Code required to reproduce the findings of our work is included in this folder. Scripts are written primarily in Python, with R used for the empirical regression analysis. The scripts titled `Fig1.ipynb`, `Fig2.ipynb`, etc. contain the final code that produces the main text figures and most of the numbers cited in the text. 

- **Figures/**: Figures are saved here. This is also where figures will be saved if you run the scripts. Minor postprocessing in Adobe Illustrator was used occasionally (e.g., for Figure 2) but generally the figures output by the scripts will be the same as what appears in the paper.


### Data 

Datasets not available in this repository are publicly available as follows:

- The **HadISST** sea surface temperature data are available from the Hadley Centre [here](https://www.metoffice.gov.uk/hadobs/hadisst/).

- The **Berkeley Earth** surface temperature data are available [here](https://berkeleyearth.org/data/). Our analysis uses the combined monthly land and ocean dataset at a 1-degree resolution.

- The **GPCC** precipitation data are available [here](https://psl.noaa.gov/data/gridded/data.gpcc.html). Our analysis the "v2020" monthly total precipitation data.

- **CMIP6 temperature, precipitation, and SST** data are generally available from the [Earth System Grid Federation](https://esgf-node.llnl.gov/search/cmip6/). Our analysis uses monthly temperature ("tas_Amon"), daily precip ("pr_day"), and monthly SST ("tos_Omon") data from as many models as were available in ~fall 2021 for SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5. Tables S3-S6 in the Supplementary Information show the models we use, along with the number of realizations from each model. 

### Scripts

Each script performs a specific step of the analysis as follows:

- `Observed_ENSO_Indices.ipynb` reads the observational (HadISST) data and calculates the E- and C-index, as well as the Ni\~{n}o3 and Ni\~{n}o3.4 ENSO indices.

