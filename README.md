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
- add python script for running all configurations
- run release-debug (assertions enabled) on wildstyle, release on titan + summit
- run all problems with 0.1.0 (then commit) and 0.1.1 (then commit)
- add spack lock files from environments where they're being run?
- add tags for each celeritas tag where the versions are run

## Directory structure

- `input/`: input detector geometry and hepmc3 files
- `results/{reldeb,opt}-{crusher,summit,wildstyle}`: output JSON files

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

In these in initial tests, the field was defined to have a value of `[0,0,1]`.

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

# System environments

## Summit

Summit uses a chained installation of Spack.
- [Chained installation
  notes](https://docs.olcf.ornl.gov/software/spack_env/summit_spack_env.html#getting-started)
  are out of date
- Spack environment used to build summit toolchains is at
  `/sw/sources/facility-spack/summit/hosts/summit/envs/base/spack.yaml`
- Spack source directory is `/sw/sources/facility-spack/summit/spack`

Key locations:

- Spack repository: `/ccs/proj/csc404/spack` including environment files
- Spack installation and view: `$PROJWORK/csc404/celeritas/spack`
- Celeritas builds: `/gpfs/alpine/proj-shared/csc404/celeritas/build*`
- Regression repo: `$HOME/celeritas-regression`

See [Summit guide](https://docs.olcf.ornl.gov/systems/summit_user_guide.html)
for more details.

## Frontier
