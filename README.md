# Celeritas regression repository

This repository is for managing the history of a suite of test problems in
Celeritas to track whether the code is able to run to completion without
hitting an assertion, how the code input options (and processed output) change
over time, and how the kernel occupancy requirements change in response to
growing code complexity.  For more detailed analysis of performance and
accuracy, see the [benchmarks
repo](https://github.com/celeritas-project/benchmarks).

The table is derived from https://github.com/celeritas-project/celeritas/issues/460 .

# Running the code

TODO:
- add input files
- add python script for running all configurations
- run debug on wildstyle, release on titan + summit
- add spack lock files from environments where they're being run?

# Table of results

Problem | EM + Physics | Geometry | Architecture | Primaries | Status (version)
-- | -- | -- | -- | -- | --
TestEM15 | MSC | ORANGE | CPU | 10k |  ✅ (v0.1.0)
TestEM15 | Field | ORANGE | CPU | 10k |   ✅ (v0.1.0)
TestEM15 | Field + MSC | VecGeom | CPU | 10k |   ✅ (v0.1.0)
TestEM15 | Field + MSC | ORANGE | CUDA | 1m |   ✅ (v0.1.0)
Simple CMS | MSC | ORANGE | CPU | 10k |  ✅ (v0.1.1)
Simple CMS | Field | ORANGE | CPU | 10k |  ✅ (v0.1.1)
Simple CMS | Field + MSC | ORANGE | CPU | 10k |  4
Simple CMS | Field + MSC | VecGeom | CUDA | 500k |  7
TestEM3 | Field | ORANGE | CPU | 10k |  4
TestEM3 | MSC | ORANGE | CUDA | 200k |  ✔️
TestEM3 | Field + MSC | VecGeom | CUDA | 200k |  6, 7
TestEM3 | — | ORANGE | HIP | 200k |  
TestEM3 | Field + MSC | ORANGE | HIP | 200k |  
CMS2018 | — | VecGeom | CPU | 5k |  ✔️
CMS2018 | — | VecGeom | CUDA | 50k |  6, then ✔️
CMS2018 | Field | VecGeom | CPU | 2k |  6
CMS2018 | Field + MSC | VecGeom | CPU | 1k |  7
CMS2018 | Field + MSC | VecGeom | CUDA | 20k |  6

For the 'version', use `git describe --tags --match 'v*'`, and please make sure
you're on the `master`/`main` branch!

## Failing assertions:

1.
(fixed by celeritas-project/celeritas#501)
```
src/orange/OrangeTrackView.hh:378:
celeritas: internal assertion failed: init.volume
```
2.
```
src/celeritas/field/FieldDriver.hh:203:
celeritas: internal assertion failed: succeeded
```
3.
```
src/celeritas/field/MagFieldEquation.hh:97:
celeritas: internal assertion failed: momentum_mag2 > 0
```
4.
```
src/celeritas/track/detail/ProcessSecondariesLauncher.hh:94:
celeritas: internal assertion failed: !geo.is_on_boundary()
```
5.
```
src/orange/OrangeTrackView.hh:363:
celeritas: internal assertion failed: this->surface_id()
```
6.
(shows up more with 56 bins per decade, less with 7)
```
src/celeritas/global/alongstep/AlongStep.hh:122:
celeritas: internal assertion failed: mfp > 0
```
7.
(fixed by celeritas-project/celeritas#499)
```
src/celeritas/global/alongstep/AlongStep.hh:83:
celeritas: internal assertion failed: p.distance < local.step_limit.step
```
8.
(fixed by celeritas-project/celeritas#490)
```
src/celeritas/em/interactor/MollerBhabhaInteractor.hh:109:
celeritas: internal assertion failed: electron_cutoff_ >= value_as<Energy>(shared_.min_valid_energy())
```
