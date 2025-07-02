#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm.auto import tqdm

# Custom libraries for working with spatial grids and KDEs
import quadgrid as qg
import kdetools as kt

# Load ERSSTv5 class for SST processing
from .ersstv5 import ERSSTv5


class Tracker():
    def __init__(self, urg_res=2):
        """Main class for generating synthetic tropical cyclone tracks.

        Parameters
        ----------
        urg_res : float
            Resolution of uniform quadgrid over which tracks will be modelled.
        url : str, optional
            URL of Copernicus Data Store.
        """

        self.basins = ['EP','NA','NI','SA','SI','SP','WP']
        self.agencies = ['WMO','USA','TOKYO','CMA','HKO','KMA','NEWDELHI',
                         'REUNION','BOM','NADI','WELLINGTON']
        self.agency_map = {'NA': 'USA', 'WP': 'TOKYO', 'NI': 'NEWDELHI',
                           'SI': 'BOM', 'SP': 'BOM', 'EP': 'USA', 'SA': 'WMO'}
        self.basin_bounds = {'NA': {'lon_bnds': [-110, 10], 'lat_bnds': [5, 60]},
                             'WP': {'lon_bnds': [80, 180], 'lat_bnds': [5, 60]},
                             'NI': {'lon_bnds': [45, 160], 'lat_bnds': [0, 40]},
                             'SI': {'lon_bnds': [25, 180], 'lat_bnds': [0, -50]},
                             'EP': {'lon_bnds': [-180, -50], 'lat_bnds': [5, 50]},
                             'SP': {'lon_bnds': [100, 180], 'lat_bnds': [-5, -55]},
                             'SA': {'lon_bnds': [-50, -30], 'lat_bnds': [-20, -40]},}

        # Initialise the global grid and define BallTree for fast lookup
        urg = qg.QuadGrid(urg_res)
        urg_pd = urg.to_pandas().reset_index().sort_values('qid')
        self.qids = urg_pd['qid'].values
        self.urg_btree = BallTree(np.deg2rad(urg_pd[['lat','lon']]), metric='haversine')
        self.urg_res = urg_res

        # IBTrACS preparation
        self.ibtracs_url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/'
        self.ibtracs = {}

        # Get current directory, for saving data
        self.cwd = Path(__file__).parent.resolve()

        # Load Natural Earth land shapefile which comes with the package
        self.land = gpd.read_file(self.cwd/'../data/ne_10m_land/ne_10m_land.shp')

    def get_ibtracs(self, basin, agency, version='v04r01', outpath=None, url=None):
        """Function to retrieve and preprocess raw IBTrACS track data.
        """

        # Input validation
        if basin.upper() not in self.basins:
            print(f'Invalid basin; must be one of {",".join(self.basins)}')
            return None
        if agency.upper() not in self.agencies:
            print(f'Invalid agency; must be one of {",".join(self.agencies)}')
            return None

        # Download data from web
        fname = f'ibtracs.{basin.upper()}.list.{version}.csv'
        if url is not None:
            file_url = url+fname
        else:
            file_url = self.ibtracs_url+fname

        print(f'Downloading IBTrACS data from {file_url}')

        # Define column to use for central pressure
        pres_col = f'{agency.upper()}_PRES'
        cols = ['SID','SEASON','NAME','ISO_TIME','NATURE','LON','LAT',pres_col]
        dtypes = {'SID': str, 'SEASON': np.int32, 'NAME': str, 'NATURE': str,
                  'LAT': np.float32, 'LON': np.float32}
        converters = {pres_col: lambda x: np.nan if x == ' ' else np.float32(x)}
        ibtracs = pd.read_csv(file_url, skiprows=[1], usecols=cols,
                              dtype=dtypes, converters=converters, parse_dates=['ISO_TIME'])

        # Preprocess
        ibtracs['qid'] = qg.lls2qids(ibtracs['LON'], ibtracs['LAT'], res_target=self.urg_res)
        ibtracs['year'] = ibtracs['ISO_TIME'].dt.year
        ibtracs['month'] = ibtracs['ISO_TIME'].dt.month
        self.ibtracs[basin.upper()] = ibtracs

        # Save to disk
        fname = f'ibtracs.{basin.upper()}.{agency}.{version}.csv'
        if outpath is None:
            outpath = (self.cwd / '../data/ibtracs/').resolve()
        ibtracs.to_csv(f'{outpath}/{fname}', index=False)
        print(f'Data saved to {outpath}')

    def load_ibtracs(self, basin, agency, inpath=None, version='v04r01'):
        """Load processed IBTrACS data from disk.
        """
        fname = f'ibtracs.{basin.upper()}.{agency}.{version}.csv'
        if inpath is None:
            inpath = (self.cwd / '../data/ibtracs').resolve()
        self.ibtracs[basin.upper()] = pd.read_csv(f'{inpath}/{fname}', parse_dates=['ISO_TIME'])

    def get_roni(self, outpath=None):
        """Download and process the Relative Oceanic Nino Index (RONI) from NOAA.
        """

        seas2month = {'DJF':2,'JFM':3,'FMA':4,'MAM':5,'AMJ':6,'MJJ':7,
                    'JJA':8,'JAS':9,'ASO':10,'SON':11,'OND':12,'NDJ':1}
        roni_raw = pd.read_csv('https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt', sep='\\s+')
        roni_raw['month'] = roni_raw['SEAS'].map(seas2month)
        roni = roni_raw.drop('SEAS', axis=1).rename(columns={'YR':'year','ANOM':'roni'}
                                                    )[['year','month','roni']]
        if outpath is None:
            outpath = (self.cwd / '../data/enso/').resolve()
        roni.to_csv(f'{outpath}/roni.csv', index=False)
        self.roni = roni

    def load_roni(self, inpath=None, fname=None):
        """Load processed Relative Oceanic Nino Index (RONI) data from disk.
        """
        if inpath is None:
            inpath = (self.cwd / '../data/enso').resolve()
        if fname is None:
            fname = 'roni.csv'
        self.roni = pd.read_csv(f'{inpath}/{fname}')

    def get_oni(self, outpath=None):
        """Download and process the Oceanic Nino Index (ONI) from NOAA.
        """

        oni_raw = pd.read_csv('https://psl.noaa.gov/data/correlation/oni.data',
                              skiprows=1, header=None, skipfooter=8, index_col=0,
                              na_values='-99.9', sep='\\s+', engine='python')
        oni = oni_raw.rename_axis('year').rename_axis('month', axis=1
                                                      ).stack().rename('oni').reset_index()
        if outpath is None:
            outpath = (self.cwd / '../data/enso/').resolve()
        oni.to_csv(f'{outpath}/oni.csv', index=False)
        self.oni = oni

    def load_oni(self, inpath=None, fname=None):
        """Load processed Oceanic Nino Index (ONI) data from disk.
        """
        if inpath is None:
            inpath = (self.cwd / '../data/enso').resolve()
        if fname is None:
            fname = 'oni.csv'
        self.oni = pd.read_csv(f'{inpath}/{fname}')

    def load_SSTs(self, inpath=None, year_range=(None, None)):
        """Load multiple ERSSTv5 files and convert for lookup.

        Parameters
        ----------
            inpath : str
                Path to ERSSTv5 monthly NetCDF files.
            year_range : (int, int), optional
                Year range to process.

        Returns
        -------
            da : DataArray
                Processed DataArray.
        """

        if inpath is None:
            inpath = (self.cwd / '../data/ersstv5').resolve()

        ersstv5 = ERSSTv5()
        self.ssts = ersstv5.load(inpath, year_range)

    def make_design_matrix(self, basin, year_range=(1950, None),
                           enso_col='roni', natures=['TS']):
        """Feature engineering step to assemble design matrix from IBTrACS
        data, ENSO features, SSTs and other environmental predictors.

        Parameters
        ----------
            basin : str
                Basin to process. The basin data must have already been loaded
                using the load_ibtracs() method.
            year_range : (int, int), optional
                Year range to process.
            enso_col : str, optional
                ENSO feature to use. Options are 'roni' (default) or 'oni'.
            natures : [str], optional
                List of 'NATURE' values to include. Options include some or all
                of: TS, ET, MX, SS, NR, DS. Defaults to TS (Tropical Storm).
                Refer to IBTrACS column documentation for more information.

        Returns
        -------
            da : DataArray
                Processed DataArray.
        """

        year_from, year_to = year_range
        ibtracs_basin = self.ibtracs.get(basin.upper(), None)

        if ibtracs_basin is None:
            print(f'IBTrACS data for basin {basin.upper()} not loaded')
            return None

        # Identify pressure column
        pres_col = [col for col in ibtracs_basin.columns if 'PRES' in col][0]

        # Filter basin data to target year range and natures
        year_mask = (ibtracs_basin['SEASON']>=year_from) & (ibtracs_basin['SEASON']<=year_to)
        subset = ibtracs_basin[year_mask & ibtracs_basin['NATURE'].isin(natures)]

        # Combine track data with exogenous environmental forcings
        # ENSO index
        enso = self.roni if enso_col.lower() == 'roni' else self.oni
        subset = subset.merge(enso, on=['year','month'])
        subset[enso_col] += np.random.normal(0, 0.0001, size=subset.shape[0])

        # Local SSTs
        year_da = xr.DataArray(subset['year'].values)
        month_da = xr.DataArray(subset['month'].values)
        lat_da = xr.DataArray(subset['LAT'].values)
        lon_da = xr.DataArray(subset['LON'].values)
        subset['sst'] = self.ssts.sel(year=year_da, month=month_da,
                                      latitude=lat_da, longitude=lon_da, method='nearest').values

        # Create time lags
        cols = ['LON','LAT',pres_col]
        X = []
        for _, sid_df in tqdm(subset.groupby('SID')):
            df = pd.concat([sid_df[enso_col.lower()],
                            sid_df['sst'],
                            sid_df[cols].shift(1).add_suffix('-1'),
                            sid_df[cols],
                            sid_df[cols].shift(-1).add_suffix('+1')], axis=1)

            # Select successive timesteps only if the differences are 3 hours
            delta1_mask = (sid_df['ISO_TIME'] - sid_df['ISO_TIME'].shift(1)) == pd.Timedelta('3h')
            delta2_mask = (sid_df['ISO_TIME'].shift(-1) - sid_df['ISO_TIME']) == pd.Timedelta('3h')
            df = df[delta1_mask & delta2_mask].dropna(subset=[f'{pres_col}-1',f'{pres_col}+1','LON-1','LON+1'])
            if df.shape[0] > 0:
                X.append(df)

        # Combine sequences into a DataFrame, adding white noise to SSTs for
        X = pd.concat(X).fillna({'sst': 0}).reset_index(drop=True)
        X['sst'] += np.random.normal(0, 0.0001, size=X.shape[0])
        return X

    def fit(self, X, bw_method='silverman', k=9, s=200, min_recs=50, wts=True):
        """Fit KDE autoregression models for all cells in a basin.

        Parameters
        ----------
            X : DataFrame
                Design matrix with all features.
            bw_method : str, optional
                Bandwidth method used for KDE fitting. Options are 'silverman',
                'scott' and 'cv'.
            k : int, optional
                Size of neighbourhood around an individual cell when combining
                records from the design matrix for model-fitting. For symmetry,
                should be the square of an odd number, i.e. 1, 9, 25, etc.
            s : float, optional
                Standard deviation of Gaussian weight kernel for neighbourhood.
                Defaults to 200 km.
            min_recs : int, optional
                Minimum number of records to fit a model. Defaults to 50.
            wts : bool, optional
                Weight data by distance from qid centroid. Defaults to True.

        Returns
        -------
            models : dict
                Dictionary of fitted KDE models.
        """

        self.models, self.missing = {}, []

        # Array of all qids in the design matrix
        qids = qg.lls2qids(X['LON'].values, X['LAT'].values, self.urg_res)

        # Loop over each qid cell in turn to fit qid-level models
        for qid in tqdm(np.unique(qids)):
            # Define cell neighbourhood around current qid
            lon, lat = qg.qid2ll(qid, self.urg_res)
            d, i = self.urg_btree.query(np.deg2rad([lat, lon])[None,:], k=k)

            # Subset training data to qids in current neighbourhood
            X_qid = X[np.isin(qids, self.qids[i])]

            # Weights as a function of distances from each point in neighbourhood to qid centroid
            dists = qg.dmat(np.array([lon]), np.array([lat]),
                            X_qid['LON'].values, X_qid['LAT'].values).ravel()
            weights = np.exp(-(dists/s)**2)

            # If enough points exist, fit model, otherwise treat qid as "missing"
            if X_qid.shape[0]>=min_recs:
                self.models[qid] = kt.gaussian_kde(X_qid.T.values,
                                                   bw_method=bw_method,
                                                   weights=weights if wts else None)
                # For bw_method='cv', track unconverged models
                if bw_method == 'cv' and not self.models[qid].res['success']:
                    print(f'Warning: {qid} model did not converge')
            else:
                self.missing.append(qid)

        return self.models

    def simulate(self, lon0, lat0, pres0, lon1, lat1, pres1, enso, ssts, m,
                 n=320, end_pres=1010):
        """Simulate ensembles of stochastic tracks.

        Parameters
        ----------
            lon0 : float
                Longitude at t=0.
            lat0 : float
                Latitude at t=0.
            pres0 : float
                Central pressure at t=0.
            lon1 : float
                Longitude at t=1.
            lat1 : float
                Latitude at t=1.
            pres1 : float
                Central pressure at t=1.
            enso : float
                ENSO index value, consistent with the index (ONI or RONI)
                the models were trained on.
            ssts : DataArray
                2D DataArray of SSTs with dimensions longitude and latitude,
                consistent with the month the TC is being simulated in.
            m : int
                Number of ensemble members to simulate.
            n : int, optional
                Maximum number of 3-hour steps. Defaults to 320 (40 days).
            end_pres : float, optional
                Central pressure at which simulated tracks stop.
                Defaults to 1010 hPa.

        Returns
        -------
            tracks_df : DataFrame
                Track ensemble at 3-hourly intervals with longitude, latitude
                and central pressure.
        """

        # Initialise arrays
        tracks = np.full((m, n, 3), np.nan)
        active = np.full(m, True)
        tracks[:,0] = [lon0, lat0, pres0]
        tracks[:,1] = [lon1, lat1, pres1]

        # Land lon, lat; for terminated tracks, and DataArray.sel doesn't work with nans
        lon_land, lat_land = 10.624, -61.315

        # Loop over time steps
        for i in tqdm(range(2, n)):
            # Identify all the qids the ensemble members are in
            track_qids = qg.lls2qids(tracks[:,i-1,0], tracks[:,i-1,1], self.urg_res)

            # Identify SSTs for all ensemble members
            lons, lats = tracks[:,i-1,:2].T
            lons_da = xr.DataArray(lons).fillna(lon_land)
            lats_da = xr.DataArray(lats).fillna(lat_land)
            track_ssts = np.nan_to_num(ssts.sel(longitude=lons_da, latitude=lats_da,
                                                method='nearest', tolerance=self.urg_res))

            # Loop over ensemble members and simulate from autoregressive models
            for j, track_qid in enumerate(track_qids):
                if track_qid not in self.models or not active[j]:
                    # Track is out of the domain or has been terminated
                    continue
                else:
                    x_cond = np.hstack([np.array([enso])[None,:],
                                        np.array([track_ssts[j]])[None,:],
                                        tracks[j,i-2][None,:],
                                        tracks[j,i-1][None,:]])

                    # Key conditional sampling step
                    tracks[j,i,:] = self.models[track_qid].conditional_resample(1, x_cond, range(x_cond.shape[1]))

                    # Track termination checks
                    if tracks[j,i,-1] >= end_pres:
                        active[j] = False
                    else:
                        continue
        # Convert numpy array to pandas format
        tracks_df = xr.DataArray(tracks, coords={'n': np.arange(tracks.shape[0]),
                                                 't': np.arange(tracks.shape[1]),
                                                 '': ['LON','LAT','PRES']}
                                                 ).to_series().dropna().unstack('')
        return tracks_df

    def resimulate(self, sid, basin, enso=None, ssts=None, m=100, n=320,
                   end_pres=1010, start_ix=0):
        """Convenience function to re-simulate ensembles of a historic track.

        Parameters
        ----------
            sid : str
                IBTrACS SID.
            basin : str
                Basin abbreviation.
            enso : float, optional
                ENSO index value, consistent with the index (ONI or RONI)
                the models were trained on.
            ssts : DataArray, optional
                2D DataArray of SSTs with dimensions longitude and latitude,
                consistent with the month the TC is being simulated in.
            m : int, optional
                Number of ensemble members to simulate.
            n : int, optional
                Maximum number of 3-hour steps. Defaults to 320 (40 days).
            end_pres : float, optional
                Central pressure at which simulated tracks stop.
                Defaults to 1010 hPa.
            start_ix : int, optional
                Timestep index to start simulation from. Defaults to 0.

        Returns
        -------
            tracks_df : DataFrame
                Track ensemble at 3-hourly intervals with longitude, latitude
                and central pressure.
        """

        agency = self.agency_map[basin.upper()]

        # Identify historical track and extract initial conditions
        data = self.ibtracs[basin.upper()]
        data_sid = data[data['SID']==sid]
        year, month = data_sid['year'].min(), data_sid['month'].min()
        lon0, lat0, pres0 = data_sid.iloc[start_ix][['LON','LAT',f'{agency}_PRES']]
        lon1, lat1, pres1 = data_sid.iloc[start_ix+1][['LON','LAT',f'{agency}_PRES']]

        if enso is None:
            enso = self.roni[(self.roni['year']==year)&(self.roni['month']==month)]['roni'].sum()
        if ssts is None:
            ssts = self.ssts.sel(year=year, month=month)
        tracks_df = self.simulate(lon0, lat0, pres0, lon1, lat1, pres1,
                                  enso, ssts, m, n, end_pres)
        return tracks_df

    def plot_missing_qids(self, figsize=(8,6)):
        """Plot all qids for which no models have been fitted because
        insufficient data is available.
        """

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.land.plot(ax=ax, color='0.5')
        ax.scatter(*qg.qids2lls(self.missing, self.urg_res), s=3, color='r')
        ax.axhline(0, lw=0.25, color='0.5')
        return fig

    def plot_tracks(self, tracks_df, basin, density=True, ax=None, figsize=(18,9)):
        """Visualise tracks and track density.
        """

        # Number of ensemble members
        m = tracks_df.index.levels[0].size

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax2 = inset_axes(ax, width='20%', height=1.2, loc='upper right')

        self.land.plot(ax=ax, lw=1, color='0.5')
        for i in range(m):
            tracks_df.loc[i].plot(x='LON', y='LAT', color='k', lw=0.25, legend=False, ax=ax)
            tracks_df.loc[i]['PRES'].plot(ax=ax2, color='0.25', lw=0.25, alpha=0.5)

        #tr.ibtracs['NA'][tr.ibtracs['NA']['SID']==sid].plot(x='LON', y='LAT', color='r', lw=2, legend=False, ax=ax)
        ax.axhline(0, lw=0.25, color='0.5')
        ax.set_xlim(self.basin_bounds[basin.upper()]['lon_bnds'])
        ax.set_ylim(self.basin_bounds[basin.upper()]['lat_bnds'])
        ax2.set_xticklabels([]); ax2.set_ylabel('Pressure'); ax2.set_ylim((850,1020)); ax2.grid(lw=0.1);

        return None
