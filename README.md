
The codebase of the diploma thesis: Simulation-based estimation methods in financial econometrics: Analysis of performance and comparison

### Folder `src`

The source code of the diploma thesis.

- `main.jl` the main function which is called by the individual scripts, handling all the simulation variants
- `ABC` implementation of the ABC-SMC algorithm with regression adjustment based on Lux (2023)
- `Bayes` implementation of the likelihood based-SMC algorithm based on Zhang et al. (2023)
- `NPMSLE` main functions of the NPMSLE (likelihood calculation is inside utils) based on Kukacka & Barunik (2017), Kristensen & Shin
(2012) and Creel & Kristensen (2012).
- `SMM` SMM algorithm provided by Zila & Kukacka (2023), thank you for that! (moment calculation is inside utils)
- `utils` contains code which is used across methods: data related functions, likelihood approximation, particle initialization, moments and weighting matrix related function, noise generation for likelihood methods, saving results
- `models` implementation of the individual models

### Folder `scripts`

The `scripts` folder contains the startup scripts of individual simulation exercises.

### Folder `results`

The `plots` folder contains resulting `*.jld` files with results from given simulation exercises.
Note, that `traj_*.jld` were created seperately once to get info about the characteristic of the likelihood approximated using trajectory.

### Folder `ntbks`

The `ntbks` folder contains Jupyter notebooks for analyzing individual results, mainly calculating RMSE, creating tables and density plots.

### Folder `plots`

The `plots` folder contains resulting graphs used in diploma thesis.

