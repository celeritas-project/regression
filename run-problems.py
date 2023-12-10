#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
- Loop over all problems
- Launch simultaneously on multiple cores (different seed per run!)
- Save overall times from all runs, and output from one run
- Catch failure message and save

Requires Python 3.7+.
"""

import asyncio
import itertools
import math
import json
from pathlib import Path, PurePath
from pprint import pprint
from os import environ
import shutil
from signal import SIGINT, SIGTERM, SIGKILL
import subprocess
import sys
import time

from summarize import inp_to_nametuple, summarize_all, exception_to_dict, get_num_events_and_primaries

systems = {}

class System:
    name = None
    build_dirs = {}
    num_jobs = None # Number of simultaneous jobs to run
    gpu_per_job = None
    cpu_per_job = None

    def get_runtime_environ(self, inp):
        env = {}

        omp_threads = self.cpu_per_job
        if inp['use_device']:
            omp_threads = 1
        else:
            env['CELER_DISABLE_DEVICE'] = "1"

        if not inp['_use_celeritas']:
            assert inp['_exe'] == "celer-g4"
            env['CELER_DISABLE'] = "1"

        if inp['_exe'] == "celer-g4":
            # Let Geant4 handle the threading
            omp_threads = 1
            env['G4FORCE_RUN_MANAGER_TYPE'] = "MT"
            env['G4FORCENUMBEROFTHREADS'] = str(self.cpu_per_job)
        else:
            assert inp['_exe'] == "celer-sim"

        env['OMP_NUM_THREADS'] = str(omp_threads)
        return env

    def create_celer_subprocess(self, inp):
        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")
        cmd = build / "bin" / inp['_exe']

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))
        if inp['use_device']:
            env['CUDA_VISIBLE_DEVICES'] = str(inp['_instance'])

        return asyncio.create_subprocess_exec(
            cmd, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    def get_monitoring_coro(self):
        return []

class Wildstyle(System):
    build_dirs = {
        'orange': Path("/home/s3j/.local/src/celeritas/build-reldeb"),
        'vecgeom': Path("/home/s3j/.local/src/celeritas/build-reldeb-vecgeom"),
    }
    name = "wildstyle"
    num_jobs = 2
    gpu_per_job = 1
    cpu_per_job = 32


class Local(System):
    build_dirs = {
        "orange": Path("/Users/seth/Code/celeritas-temp/build"),
    }
    name = "testing"
    num_jobs = 1
    gpu_per_job = 0
    cpu_per_job = 1


class Summit(System):
    _CELER_ROOT = Path(environ.get('PROJWORK', '')) / 'csc404' / 'celeritas'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug-novg',
        "vecgeom": _CELER_ROOT / 'build-ndebug',
    }
    name = "summit"
    num_jobs = 6
    gpu_per_job = 1
    cpu_per_job = 7

    def create_celer_subprocess(self, inp):
        cmd = "jsrun"
        env = dict(environ)
        env.update(self.get_runtime_environ(inp))

        args = [
            "-n1", # total resource sets
            "-r1", # resource sets per host
            "-a1", # tasks per resource set
            f"-c{self.cpu_per_job}", # CPUs per resource set
            "--bind=packed:7",
            "--launch_distribution=packed",
        ]
        if inp['use_device']:
            args.append("-g1") # GPUs per resource set
        else:
            args.append("-g0")

        args.extend("".join(["-E", k, "=", v]) for k, v in env.items())

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")

        args.extend([
            build / "bin" / inp['_exe'],
            "-"
        ])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def run_jslist(self):
        # Wait a second for the jobs to start
        await asyncio.sleep(1)
        print("Running jslist")

        try:
            proc = await asyncio.create_subprocess_exec("jslist", "-r", "-R")
        except FileNotFoundError as e:
            print("jslist not found :(")
            return

        print("Waiting on jslist output")
        await proc.communicate()

    def get_monitoring_coro(self):
        return [self.run_jslist()]

class Frontier(System):
    _CELER_ROOT = Path(environ['HOME']) / '.local' / 'src' / 'celeritas-frontier'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug'
    }
    name = "frontier"
    num_jobs = 8
    gpu_per_job = 1
    cpu_per_job = 7

    # NOTE: layout multi-gpu run
    # num_jobs = 4
    # gpu_per_job = 2
    # cpu_per_job = 14

    def create_celer_subprocess(self, inp):
        cmd = "srun"

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))

        args = [
            f"--cpus-per-task={self.cpu_per_job}",
        ]
        if inp['use_device']:
            args.append("--gpus-per-task=1")
            args.append("--gpus=0")
        else:
            args.append("--gpus=0")

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            raise RuntimeError("Geometry type unavailable")

        exe = build / "bin" / inp['_exe']
        if not exe.exists():
            raise FileNotFoundError(exe)
        args.extend([str(exe), "-"])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

class Crusher(Frontier):
    _CELER_ROOT = Path(environ['HOME']) / '.local' / 'src' / 'celeritas-crusher'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug'
    }
    name = "crusher"


class Perlmutter(Frontier):
    # System details:
    # https://portal.nersc.gov/cfs/mpccc/sleak/userdocs4/systems/perlmutter/system_details/#system-specification-phase-1
    _CELER_ROOT = Path(environ.get('CFS', '')) / 'atlas' / 'esseivaj' / 'devel' / 'celeritas'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug-novg',
        "vecgeom": _CELER_ROOT / 'build-ndebug',
    }
    name = "perlmutter"
    num_jobs = 4 # Nvidia A100 per node
    gpu_per_job = 1
    cpu_per_job = 16 # 1/4 of AMD EPYC with no hyperthreading

    def create_celer_subprocess(self, inp):
        cmd = "srun"

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))
        # number of virtual CPUS
        n_cpus = int(2 * (64 / self.num_jobs))

        # 2 hyperthreads per core on Perlmutter
        assert self.cpu_per_job * 2 == n_cpus
        args = [
            f"--cpus-per-task={n_cpus}",
            "--ntasks=1",
            "--cpu-bind=verbose,cores"
        ]
        if inp['use_device']:
            args.append("--gpus-per-task=1")
            args.append("--gpu-bind=verbose,closest")
        else:
            args.append("--gpus=0")

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")

        exe = build / "bin" / inp['_exe']
        if not exe.exists():
            raise FileNotFoundError(exe)
        args.extend([str(exe), "-"])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

regression_dir = Path(__file__).parent
input_dir = regression_dir / "input"

base_input = {
    "_geometry": "orange",
    "_exe": "celer-sim",
    "_timeout": 600.0,
    "_use_celeritas": True,
    "use_device": False,
    "merge_events": False,
    "sync": False,
    "initializer_capacity": 2**24,
    "num_track_slots": 2**16,
    "max_steps": 2**21,
    "secondary_stack_factor": 3.0,
    "brem_combined": False,
    "physics_options": {
        "coulomb_scattering": False,
        "rayleigh_scattering": False,
        "eloss_fluctuation": True,
        "lpm": True,
        "em_bins_per_decade": 56,
        "physics": "em_basic",
        "msc": "none",
    },
    "primary_options": {
        "seed": 0,
        "pdg": 11,
        "energy": 10000,  # 10 GeV
        "position": [0, 0, 0],
        "direction": {"distribution": "isotropic"},
        "primaries_per_event": 1300,  # 13 TeV
    },
}

use_geant = {
    "_exe": "celer-g4",
    "merge_events": False, # can't actually merge IRL
    "physics_list": "geant_physics_list",
    "sd_type": "none",
    "output_file": "-",
}

pure_geant = {
    "_geometry": "geant4",
    "_use_celeritas": False,
    "physics_options": {
        # Since geant4 uses splines it doesn't need as many points
        "em_bins_per_decade": 14,
    }
}

use_msc = {"physics_options": {"msc": "urban"}}
use_field = {
    "field": [0.0, 0.0, 1.0],
}

use_gpu = {
    "use_device": True,
    "merge_events": True,
    "num_track_slots": 2**20,
    "max_steps": 2**15,
    "initializer_capacity": 2**26,
}

use_sync = {
    "sync": True,
}

testem15 = {
    "geometry_file": "testem15.gdml",
    "primary_options": {
        "pdg": [11, -11],
    },
    "sync": False,
}

simple_cms = {
    "geometry_file": "simple-cms.gdml",
}

testem3 = {
    "geometry_file": "testem3-flat.gdml",
    "primary_options": {
        "position": [-22, 0, 0],
        "direction": [1, 0, 0],
    }
}

full_cms = {
    "_geometry": "vecgeom",
    "geometry_file": "cms2018.gdml",
    "cuda_stack_size": 8192,
}

use_vecgeom = {"_geometry": "vecgeom"}

# List of list of setting dictionaries
problems = [
    [testem15],
    [testem15, use_field],
    [testem15, use_msc, use_field],
    [testem15, use_msc, use_field, use_vecgeom],
    [simple_cms, use_msc],
    [simple_cms, use_field],
    [simple_cms, use_field, use_msc],
    [simple_cms, use_field, use_msc, use_vecgeom],
    [testem3],
    [testem3, use_vecgeom],
    [testem3, use_field],
    [testem3, use_msc],
    [testem3, use_field, use_msc],
    [testem3, use_field, use_msc, use_vecgeom],
    [full_cms],
    [full_cms, use_field, use_msc],
]

# Run again with sync on for detailed GPU timing
sync_problems = [
    [testem15, use_field],
    [testem15, use_field, use_vecgeom],
    [testem3, use_field, use_msc],
    [testem3, use_field, use_msc, use_vecgeom],
    [full_cms, use_field, use_msc],
]

def recurse_updated(d, other):
    result = d.copy()
    result.update(other)
    for k, v in result.items():
        if isinstance(v, dict):
            try:
                orig = d[k]
            except KeyError:
                v = result[k]
            else:
                v = recurse_updated(orig, result[k])
            result[k] = v
    return result


def build_input(problem_dicts):
    """Construct an input dictionary by merging inputs.

    Later entries override earlier entries.
    """
    # Combine all dictionaries
    inp = base_input.copy()
    for d in problem_dicts:
        inp = recurse_updated(inp, d)

    # Make paths absolute
    for k in inp:
        if k.endswith('_file'):
            v = inp[k]
            if v != '-':
                inp[k] = str(input_dir / v)

    # Save name and output directory
    inp["_name"] = name = inp_to_nametuple(inp)
    inp["_outdir"] = "-".join(name)

    # Update 'maximum events' input entry
    (inp["max_events"], _) = get_num_events_and_primaries(inp)

    return inp


def build_instance(inp, instance):
    inp = inp.copy()
    inp["_instance"] = instance
    inp["seed"] = 20220904 + instance
    return inp


async def communicate_with_timeout(proc, interrupt, terminate=5.0, kill=1.0, input=None):
    """Interrupt, then terminate, then kill a process if it doesn't
    communicate.
    """
    try:
        result = await asyncio.wait_for(
            proc.communicate(input),
            timeout=interrupt)
    except asyncio.TimeoutError:
        print(f"Timed out after {interrupt} seconds: sending interrupt")
        proc.send_signal(SIGINT)
    else:
        return result

    try:
        result = await asyncio.wait_for(proc.communicate(),
                    timeout=terminate)
    except asyncio.TimeoutError:
        print(f"Timed out *AGAIN* after {terminate} seconds")
        proc.send_signal(SIGTERM)
    else:
        return result

    try:
        result = await asyncio.wait_for(proc.communicate(),
                    timeout=kill)
    except asyncio.TimeoutError:
        print(f"Set phasers to kill after {kill} seconds")
        proc.send_signal(SIGKILL)
    else:
        return result

    print("Awaiting communication")
    result = await proc.communicate()
    return result


async def run_celeritas(system, results_dir, inp):
    instance = inp['_instance']

    if not inp["merge_events"] and inp["use_device"]:
        assert inp["_exe"] == "celer-g4"
        # Round up cpu-per-job to nearest power of 2
        factor = 2**int(math.ceil(math.log2(system.cpu_per_job)))
        inp["initializer_capacity"] /= factor
        inp["num_track_slots"] /= factor

    try:
        proc = await system.create_celer_subprocess(inp)
    except Exception as e:
        print("Problem creating subprocess:", e)
        return exception_to_dict(e, context="creating subprocess")

    # TODO: monitor output, e.g. https://gist.github.com/kalebo/1e085ee36de45ffded7e5d9f857265d0

    print(f"{instance}: awaiting communcation")
    failed = False
    out, err = await communicate_with_timeout(proc,
        input=json.dumps(inp).encode(),
        interrupt=inp['_timeout']
    )

    try:
        result = json.loads(out)
    except json.decoder.JSONDecodeError as e:
        print(f"{instance}: failed to decode JSON")
        failed = True
        result = {
            'stdout': out.decode().splitlines(),
        }

    if proc.returncode:
        print(f"{instance}: exit code {proc.returncode}")
        failed = True
        result['stderr'] = err.decode().splitlines()

    # Copy special inputs to output for later processing
    result.setdefault('input', {}).update(
        {k: v for k,v in inp.items() if k.startswith('_')}
    )

    try:
        outdir = results_dir / inp['_outdir']
        outdir.mkdir(exist_ok=True)
        with open(outdir / f"{instance:d}.json", "w") as f:
            json.dump(result, f, indent=0, sort_keys=True)
    except Exception as e:
        print(f"{instance}: failed to output:", repr(e))
        failed = True

    if proc.returncode:
        # Write input to reproduce later
        with open(outdir / f"{instance:d}.inp.json", "w") as f:
            json.dump(inp, f, indent=0, sort_keys=True)

    if not failed:
        print(f"{instance}: success")

    return result


async def main():
    try:
        sysname = sys.argv[1]
    except IndexError:
        Sys = Local
    else:
        # TODO: use metaclass to build this list automatically
        _systems = {S.name: S for S in [Frontier, Summit, Crusher, Perlmutter, Wildstyle]}
        Sys = _systems[sysname]
    system = Sys()
    system.build_dirs['geant4'] = system.build_dirs['orange']

    # Copy build files
    buildfile_dir = regression_dir / 'build-files' / system.name
    buildfile_dir.mkdir(exist_ok=True)
    for k, v in system.build_dirs.items():
        shutil.copyfile(v / 'CMakeCache.txt', buildfile_dir / (k + '.txt'))

    results_dir = regression_dir / 'results' / system.name
    results_dir.mkdir(exist_ok=True)

    device_mods = []
    if system.gpu_per_job:
        device_mods.append([use_gpu])
        device_mods.append([use_gpu, use_geant])
    if True:
        # CPU-only
        device_mods.append([]) # CPU celeritas
        device_mods.append([use_geant]) # CPU celeritas through celer-g4
        device_mods.append([use_geant, pure_geant]) # CPU geant4

    # Set number of events based on number of CPUs
    base_inputs = [
        base_input,
        {"primary_options": {"num_events": system.cpu_per_job}},
    ]

    inputs = [build_input(base_inputs + p + d)
              for p, d in itertools.product(problems, device_mods)]
    inputs += [build_input(base_inputs + p + [use_gpu, use_sync])
               for p in sync_problems]
    with open(results_dir / "index.json", "w") as f:
        json.dump([(inp['_outdir'], inp['_name'])
                   for inp in inputs], f, indent=0)

    summaries = {}
    allstart = time.monotonic()
    _num_inputs = len(inputs)
    for (i, inp) in enumerate(inputs, start=1):
        print("="*79)
        name = inp['_outdir']
        print(f"Running problem {i} of {_num_inputs}: {name}...")
        start = time.monotonic()
        tasks = [run_celeritas(system, results_dir, build_instance(inp, i))
                 for i in range(system.num_jobs)]
        if not summaries:
            # Only print monitoring for first instance
            tasks.extend(system.get_monitoring_coro())
        result = await asyncio.gather(*tasks)

        # Ignore results from monitoring tasks
        result = result[:system.num_jobs]

        try:
            summaries[name] = summary = summarize_all(result)
        except Exception as e:
            print("*"*79)
            print("FAILED input:")
            pprint(inp)
            print("*"*79)
            pprint(result)
            print("Failed to summarize result above")
            raise
        summary['name'] = inp['_name'] # name tuple
        pprint(summary)
        alldelta = time.monotonic() - allstart
        delta = time.monotonic() - start
        print(f"Elapsed time for {name}: {delta:.1f} (total: {alldelta:.0f})")

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)
    print(f"Wrote summaries to {results_dir}")

asyncio.run(main())
