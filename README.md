# Tropical Cyclone Kernel Resampler (Tracker)
Tracker is an experimental Python package allowing the user to simulate synthetic Tropical Cyclone (TC) tracks using a simple statistical model fitted to historic IBTrACS data. The method is similar to various autoregressive track simulation algorithms in the literature, but is arguably simpler since it makes very few parametric assumptions, most of which can be changed. The core idea is to create a coarse grid over the ocean and for each cell use Kernel Density Estimation (KDE) to model the joint distribution of latitude, longitude and central pressure at times $t-1$, $t$ and $t+1$. Given values of latitude, longitude and central pressure at times $t-1$ and $t$, the simulation algorithm takes a conditional sample of the values at time $t+1$ from the 'current' (time $t$) cell KDE model.

Some additional features:
- Includes Sea Surface Temperature (SST) at the current grid cell as an additional predictor; a function is included to download ERSSTv5 SST data from NOAA
- Includes a monthly ENSO feature as a predictor; either ONI or RONI, downloaded from NOAA, can be used

Having ENSO and SST as predictors allows some interesting experiments to be performed:
- simple counterfactual analysis - re-run ensembles historical years under different ENSO or SST states
- stochastic set generation - when combined with a TC genesis model, this track model can be used to generate a stochastic track set. A future refinement could be to add on a simple wind field model, based on the modelled central pressure
- seasonal forecast stochastic set - when combined with a genesis forecast model and seasonal forecasts of SSTs, the model can be used to generate a dynamic stochastic set, conditional on the forecast SSTs, turning predicted basin counts into landfalling frequencies

Potential future refinements:
- Add Z500 field (geopotential height at 500 hPa) as a predictor - can try to use it to estimate steering winds
- Add vertical wind shear field

Note however, that as the dimensionality of the joint distribution increases, reliable KDE fitting becomes more difficult, and dimensional reduction techniques may need to be introduced.
