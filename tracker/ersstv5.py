#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import requests
import datetime as dt
import xarray as xr
from tqdm.auto import tqdm


class ERSSTv5():
    def __init__(self):
        """Convenience class for downloading and processing ERSSTv5 SST data.
        """

        self.url = 'https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/netcdf/'
        self.this_year = dt.datetime.now().year
        self.this_month = dt.datetime.now().month

    def download(self, outpath, year_range=(None,None), months=None,
                 overwrite=False, proxy={}):
        """Download ERSSTv5 data.

        Parameters
        ----------
            outpath : str
                Output path to save files.
            year_range : (int, int), optional
                Year range to download. Defaults to maximum possible range.
            months : list, optional
                List of months to download. Defaults to full year.
            overwrite : boolean, optional
                If True, don't check for existence of file before downloading.
                Defaults to False.
            proxy : dict, optional
                Proxy dictionary if needed.
        """

        year_from, year_to = year_range
        if year_from is None:
            year_from = 1854
        if year_to is None:
            year_to = self.this_year
        years = range(year_from, year_to+1)

        # Loop over years
        pbar = tqdm(years)
        for year in pbar:
            pbar.set_description(f'Processing {year}')
            if months is not None:
                pass
            elif year < self.this_year:
                months = range(1,13)
            else:
                months = range(1, self.this_month)
            for month in months:
                fname = f'ersst.v5.{year}{month:02}.nc'
                fpath = os.path.join(outpath, fname)
                if not overwrite and os.path.exists(fpath):
                    msg = f'Skipping {fname} as it exists'
                    pbar.set_description(f'Downloading {year} | {msg}')
                else:
                    r = requests.get(self.url+fname, stream=True, proxies=proxy)
                    if r.ok:
                        with open(fpath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024 * 8):
                                if chunk:
                                    f.write(chunk)
                                    f.flush()
                                    os.fsync(f.fileno())
                    else:  # HTTP status code 4XX/5XX
                        print(f'Download failed: '
                              f'status code {r.status_code}\n{r.text}')

    def convert(self, da):
        """Convert structure of a raw DataArray.

        Parameters
        ----------
            da : DataArray
                DataArray with dims
                ['lat','lev','lon','time'].

        Returns
        -------
            da : DataArray
                DataArray with dims ['latitude','longitude','year','month'].
        """

        # Drop unneeded dims/coords
        da = da.squeeze('lev', drop=True)

        # Convert longitudes from 0->360 to -180->180
        da['lon'] = ((da['lon'] + 180) % 360) - 180
        da = da.sortby(['lat','lon']
                       ).rename({'lat':'latitude','lon':'longitude'})

        # Convert time index to (year, month)
        da = da.assign_coords(year=('time', da.time.dt.year.data),
                              month=('time', da.time.dt.month.data)
                             ).set_index(time=('year', 'month')).unstack('time')
        return da

    def load(self, inpath, year_range=(None, None)):
        """Load and convert multiple ERSSTv5 files.

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

        year_from, year_to = year_range
        if year_from is None:
            year_from = 1854
        if year_to is None:
            year_to = self.this_year
        years = range(year_from, year_to+1)

        # Generate all file paths - assumes monthly files
        fnames = [fname for fname in os.listdir(inpath) if '.nc' in fname]

        # Check if all years requested are available in fnames
        years_fnames = [int(fname.split('.')[2][:4]) for fname in fnames]

        years_missing = sorted(set(years) - set(years_fnames))
        if len(years_missing) > 0:
            print(f'Warning: some years in year_range not in {inpath}:\n'
                  f'{", ".join(map(str, years_missing))}')

        # Generate combined DataArray for all months for this variable
        fpaths = [os.path.join(inpath, fname) for fname in sorted(fnames)
                  if int(fname.split('.')[2][:4]) in years]
        da = [self.convert(xr.open_dataset(fpath, engine='netcdf4')['sst'])
              for fpath in tqdm(fpaths)]
        return xr.combine_by_coords(da)['sst']
