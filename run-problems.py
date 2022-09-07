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
import json
from math import log10
from pathlib import Path, PurePath
from pprint import pprint
from os import environ
import re

class System:
    name = None
    build_dirs = {}
    num_jobs = None
    gpu_per_job = None
    cpu_per_job = None

    def create_celer_subprocess(self, inp):
        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")
        cmd = build / "app/demo-loop"
        env = None
        if inp['use_device']:
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
    num_jobs = 3
    gpu_per_job = 0
    cpu_per_job = 2


_fixme_systems = {
    'summit': {
        'gpu': 6,
        'cpu_per_gpu': 7,
        'orange': Path("/.../build-ndebug"),
        'vecgeom': Path("/.../build-ndebug-vecgeom"),
    },
    'crusher': {
        'gpu': 8, # NOTE: really 4 GCDs, 2 GPUs per GCD
        'cpu_per_gpu': 8, # NOTE: really 4 job nodes, so 8 procs per 2 GPU
    }
}

regression_dir = Path(__file__).parent
input_dir = regression_dir / "input"

base_input = {
    "brem_combined": False,
    "enable_diagnostics": False,
    "initializer_capacity": 1e6,
    "mag_field": [0.0, 0.0, 1.0],
    "max_num_tracks": 32,
    "max_steps": 2**30,
    "secondary_stack_factor": 3.0,
    "sync": True,
    "use_device": False,
    # Geant options
    "brem_lpm": True,
    "conv_lpm": True,
    "eloss_fluctuation": True,
    "enable_msc": False,
    "rayleigh": False,
    "geant_options": {
        "em_bins_per_decade": 56,
        "physics": "em_basic"
    },
}

use_msc = {"enable_msc": True,
           "eloss_fluctuation": False}

no_field = {
    "mag_field": [0.0, 0.0, 0.0],
}

testem15 = {
    "_geometry": "orange",
    "geometry_filename": "testem15.org.json",
    "hepmc3_filename": "testem15-10k.hepmc3",
    "physics_filename": "testem15.gdml",
    "mag_field": [0.0, 0.0, 1.0],
}

simple_cms = {
    "_geometry": "orange",
    "geometry_filename": "simple-cms.org.json",
    "hepmc3_filename": "simple-cms-10k.hepmc3",
    "physics_filename": "simple-cms.gdml",
    "mag_field": [0.0, 0.0, 1000.0],
}

testem3 = {
    "_geometry": "orange",
    "geometry_filename": "testem3.org.json",
    "physics_filename": "testem15.gdml",
    "mag_field": [0.0, 0.0, 1.0],
    "primary_gen_options": {
        "pdg": 11,
        "energy": 10000,
        "position": [-22, 0, 0],
        "direction": [1, 0, 0],
        "num_events": 10000,
        "primaries_per_event": 1
    }
}

full_cms = {
    "_geometry": "vecgeom",
    "geometry_filename": "cms2018.gdml",
    "hepmc3_filename": "simple-cms-10k.hepmc3",
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
    [simple_cms, use_msc, no_field],
    [simple_cms],
    [simple_cms, use_msc],
    [simple_cms, use_msc,
        {"_geometry": "vecgeom", "geometry_filename": "simple-cms.gdml"}],
    [testem3],
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
    return inp

def calc_emptying_step(active):
    active_it = iter(active)
    prev = next(active_it)
    max_cap = 0
    result = None
    for (i, cur) in enumerate(active_it, start=1):
        max_cap = max(prev, max_cap)
        if prev == max_cap and cur < max_cap:
            result = i
        prev = cur
    return result

def calc_queue_hwm(queued):
    hq, hi = max((q, i) for (i, q) in enumerate(queued))
    return (hi, hq)

def get_action_times(actions):
    return {a['label']: a['time'] for a in actions if a.get('time', 0) > 0}

def summarize(out):
    """Calculate statistics about the tracking behaviors.

    These are basically equivalent to those in StepperTestBase.
    """
    result = out['result']
    active = result['active']
    time = result['time']

    emptying_step = calc_emptying_step(active)
    summary = {
        "num_step_iters": len(active),
        "avg_steps_per_primary": sum(active) / active[0],
        "emptying_step": emptying_step,
        "queue_hwm": calc_queue_hwm(result['initializers']),
        "total_time": time['total'],
        "action_times": get_action_times(out['internal']['actions']),
        "pre_emptying_time": time['steps'][(emptying_step or 0) - 1]
        # occupancy if not debug?
    }
    return summary

failure_re = re.compile('(error|warning|exception|critical)', re.IGNORECASE)

def summarize_failure(out):
    try:
        return out['result']['exception']
    except KeyError:
        matches = []
        for k in ['stderr', 'stdout']:
            if k not in out:
                continue
            matches.extend(line for line in out[k]
                           if failure_re.search(line) is not None)
        return matches

async def run_celeritas(system, problem_dicts, instance):
    inp = build_input(problem_dicts)
    try:
        proc = await system.create_celer_subprocess(inp)
    except FileNotFoundError as e:
        summary = str(e)
        result = None
        return (summary, result)

    # TODO: define alarm to send process SIGINT after too long an interval
    # TODO: monitor output, e.g. https://gist.github.com/kalebo/1e085ee36de45ffded7e5d9f857265d0

    inp["seed"] = 20220904 + instance
    print(f"{instance}: awaiting communcation")
    out, err = await proc.communicate(input=json.dumps(inp).encode())

    print(f"{instance}: complete")
    try:
        result = json.loads(out)
    except json.decoder.JSONDecodeError as e:
        result = {
            'stdout': out.decode().splitlines(),
        }

    if proc.returncode:
        result['stderr'] = err.decode().splitlines()
        summary = summarize_failure(result)
    else:
        summary = summarize(result)

    return (summary, result)

async def main():
    system = Local()

    results_dir = regression_dir / 'results' / system.name
    results_dir.mkdir()

    summaries = []
    results = []
    for p in problems:
        result = await asyncio.gather(*(run_celeritas(system, p, i)
                                        for i in range(system.num_jobs)))
        [cur_summaries, cur_results] = zip(*result)
        pprint(cur_summaries)
        summaries.append(cur_summaries)
        results.append(cur_results)

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)
    with open(results_dir / 'full.json', 'w') as f:
        json.dump(results, f, indent=0, sort_keys=True)

asyncio.run(main())
