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
import json
from pathlib import Path, PurePath
from pprint import pprint
from os import environ
from signal import SIGINT, SIGTERM, SIGKILL
import subprocess
import sys

from summarize import inp_to_nametuple, summarize_all, exception_to_dict

g4env = {k: v for k, v in environ.items()
         if k.startswith('G4')}

systems = {}

class System:
    name = None
    build_dirs = {}
    num_jobs = None # Number of simultaneous jobs to run
    gpu_per_job = None
    cpu_per_job = None

    def create_celer_subprocess(self, inp):
        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")
        cmd = build / "app/demo-loop"
        env = None
        if not inp['use_device']:
            env = dict(environ)
            env['CELER_DISABLE_DEVICE'] = "1"
            env['OMP_NUM_THREADS'] = str(self.cpu_per_job)

        return asyncio.create_subprocess_exec(
            cmd, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

    def get_monitoring_coro(self):
        return []

class Wildstyle(System):
    build_dirs = {
        'orange': Path("/home/wherever/celeritas/build-reldeb-orange"),
        'vecgeom': Path("/home/wherever/celeritas/build-reldeb-vecgeom"),
    }
    name = "wildstyle"
    num_jobs = 2
    gpu_per_job = 1
    cpu_per_job = 32


class Local(System):
    build_dirs = {
        "orange": Path("/Users/seth/.local/src/celeritas/build"),
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
    num_jobs = 1
    num_jobs = 6
    gpu_per_job = 1
    cpu_per_job = 7

    def create_celer_subprocess(self, inp):
        cmd = "jsrun"
        env = g4env.copy()
        env["OMP_NUM_THREADS"] = str(self.cpu_per_job)

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
            env["CELER_DISABLE_DEVICE"] = "1"
            args.append("-g0")

        args.extend("".join(["-E", k, "=", v]) for k, v in env.items())

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")

        args.extend([
            build / "app" / "demo-loop",
            "-"
        ])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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

class Crusher(System):
    _CELER_ROOT = Path(environ['HOME']) / '.local' / 'src' / 'celeritas'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug'
    }
    name = "crusher"
    # NOTE: layout multi-gpu run
    # num_jobs = 4
    # gpu_per_job = 2
    # cpu_per_job = 16
    num_jobs = 8
    gpu_per_job = 1
    cpu_per_job = 8

    def create_celer_subprocess(self, inp):
        cmd = "srun"
        env = dict(environ)
        env["OMP_NUM_THREADS"] = str(self.cpu_per_job)

        args = [
            f"--cpus-per-task={self.cpu_per_job}",
        ]
        if inp['use_device']:
            args.append("--gpus-per-task=1")
        else:
            env["CELER_DISABLE_DEVICE"] = "1"
            args.append("--gpus=0")

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")

        args.extend([
            build / "app" / "demo-loop",
            "-"
        ])

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
    "_timeout": 600.0,
    "brem_combined": False,
    "enable_diagnostics": False,
    "initializer_capacity": 2**20,
    "mag_field": [0.0, 0.0, 1.0],
    "max_num_tracks": 2**12,
    "max_steps": 2**19,
    "secondary_stack_factor": 3.0,
    "sync": True,
    "use_device": False,
    # Geant options
    "brem_lpm": True,
    "conv_lpm": True,
    "eloss_fluctuation": False,
    "enable_msc": False,
    "rayleigh": True,
    "geant_options": {
        "em_bins_per_decade": 56,
        "physics": "em_basic"
    },
}

use_msc = {"enable_msc": True}

use_gpu = {
    "use_device": True,
    "max_num_tracks": 2**20,
    "max_steps": 2**11,
    "initializer_capacity": 2**26,
}

no_field = {
    "mag_field": [0.0, 0.0, 0.0],
    "eloss_fluctuation": True,
}

testem15 = {
    "_geometry": "orange",
    "_num_events": 7, # TODO: output this as a diagnostic
    "_num_primaries": 9100, # TODO: output this as a diagnostic
    "geometry_filename": "testem15.org.json",
    "hepmc3_filename": "testem15-13TeV.hepmc3",
    "physics_filename": "testem15.gdml",
    "mag_field": [0.0, 0.0, 1.0],
    "sync": False,
}

simple_cms = {
    "_geometry": "orange",
    "_num_events": 7, # TODO: output this as a diagnostic
    "_num_primaries": 9100, # TODO: output this as a diagnostic
    "geometry_filename": "simple-cms.org.json",
    "hepmc3_filename": "simple-cms-13TeV.hepmc3",
    "physics_filename": "simple-cms.gdml",
    "mag_field": [0.0, 0.0, 1000.0],
}

testem3 = {
    "_geometry": "orange",
    "geometry_filename": "testem3-flat.org.json",
    "physics_filename": "testem3-flat.gdml",
    "mag_field": [0.0, 0.0, 1.0],
    "sync": False,
    "primary_gen_options": {
        "pdg": 11,
        "energy": 10000,  # 10 GeV
        "position": [-22, 0, 0],
        "direction": [1, 0, 0],
        "num_events": 7,
        "primaries_per_event": 1300  # 13 TeV
    }
}

full_cms = {
    "_geometry": "vecgeom",
    "_num_events": 7, # TODO: output this as a diagnostic
    "_num_primaries": 9100, # TODO: output this as a diagnostic
    "geometry_filename": "cms2018.gdml",
    "hepmc3_filename": "simple-cms-13TeV.hepmc3",
    "physics_filename": "cms2018.gdml",
    "mag_field": [0.0, 0.0, 1000.0],
}

# List of list of setting dictionaries
problems = [
    [testem15, no_field],
    [testem15],
    [testem15, use_msc,
        {"_geometry": "vecgeom", "geometry_filename": "testem15.gdml"}],
    [testem15, use_msc],
    [simple_cms, no_field, use_msc],
    [simple_cms],
    [simple_cms, use_msc],
    [simple_cms, use_msc,
        {"_geometry": "vecgeom", "geometry_filename": "simple-cms.gdml"}],
    [testem3, no_field],
    [testem3],
    [testem3, no_field,
        {"_geometry": "vecgeom", "geometry_filename": "testem3-flat.gdml"}],
    [testem3, no_field, use_msc],
    [full_cms, no_field],
    [full_cms, use_msc],
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
    inp = base_input.copy()
    for d in problem_dicts:
        inp = recurse_updated(inp, d)
    for k in inp:
        if k.endswith('_filename'):
            inp[k] = str(input_dir / inp[k])

    inp["_name"] = name = inp_to_nametuple(inp)
    inp["_outdir"] = "-".join(name)
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


async def run_celeritas(system, results_dir, inp, instance):
    try:
        proc = await system.create_celer_subprocess(inp)
    except FileNotFoundError as e:
        return exception_to_dict(e, context="creating subprocess")

    # TODO: monitor output, e.g. https://gist.github.com/kalebo/1e085ee36de45ffded7e5d9f857265d0

    inp["seed"] = 20220904 + instance
    print(f"{instance}: awaiting communcation")
    out, err = await communicate_with_timeout(proc,
        input=json.dumps(inp).encode(),
        interrupt=inp['_timeout']
    )

    print(f"{instance}: complete")
    try:
        result = json.loads(out)
    except json.decoder.JSONDecodeError as e:
        result = {
            'stdout': out.decode().splitlines(),
        }

    if proc.returncode:
        result['stderr'] = err.decode().splitlines()

    # Copy special inputs to output for later processing
    result.setdefault('input', {}).update(
        {k: v for k,v in inp.items() if k.startswith('_')}
    )

    try:
        outdir = results_dir / inp['_outdir']
        outdir.mkdir(exist_ok=True)
        filename = f"{instance:d}.json"
        with open(outdir / filename, "w") as f:
            json.dump(result, f, indent=0, sort_keys=True)
    except Exception as e:
        print("Failed to output:", repr(e))

    return result


async def main():
    try:
        sysname = sys.argv[1]
    except IndexError:
        Sys = Local
    else:
        # TODO: use metaclass to build this list automatically
        _systems = {S.name: S for S in [Summit, Crusher, Wildstyle]}
        Sys = _systems[sysname]
    system = Sys()

    results_dir = regression_dir / 'results' / system.name
    results_dir.mkdir(exist_ok=True)

    device_mods = [[]] # CPU
    if system.gpu_per_job:
        device_mods.append([use_gpu])

    inputs = [build_input([base_input] + p + d)
              for p, d in itertools.product(problems, device_mods)]
    with open(results_dir / "index.json", "w") as f:
        json.dump([(inp['_outdir'], inp['_name'])
                   for inp in inputs], f, indent=0)

    summaries = {}
    for inp in inputs:
        print("="*79)
        # pprint(inp)
        tasks = [run_celeritas(system, results_dir, inp, i)
                 for i in range(system.num_jobs)]
        tasks.extend(system.get_monitoring_coro())
        result = await asyncio.gather(*tasks)

        # Ignore results from monitoring tasks
        result = result[:system.num_jobs]

        summaries[inp['_outdir']] = summary = summarize_all(result)
        summary['name'] = inp['_name']
        pprint(summary)

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)

asyncio.run(main())
