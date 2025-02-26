#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


# Create directory and set filename for results
results_dir = Path('results')

# Path to problem inputs
input_dir = Path("/home/alund/celeritas_project/regression/input")

# Path to build
build_dir = Path("/home/alund/celeritas_project/celeritas/build-release")

## BASE INPUT ##

base_input = {
    "_geometry": "vecgeom",
    "write_track_counts": False,
    "secondary_stack_factor": 2.0,
    "brem_combined": False,
    "physics_options": {
        "coulomb_scattering": False,
        "rayleigh_scattering": False,
        "eloss_fluctuation": True,
        "lpm": True,
        "em_bins_per_decade": 7,
        "physics": "em_basic",
        "msc": "urban",
    },
    "primary_options": {
        "seed": 0,
        "pdg": 11,
        "energy": 10000,  # 10 GeV
        "position": [0, 0, 0],
        "direction": {"distribution": "isotropic"},
        "primaries_per_event": 1300,  # 13 TeV
    },
    "use_device": True,
    "action_times": False,
    "write_track_counts": True,
    "track_order": "none",
    "merge_events": True,  # celer-sim options
    "physics_list": "celer_em",  # celer-g4 options
    "sd_type": "none",
    "output_file": "-",
}

## PHYSICS ##

use_field = {
    "field": [0.0, 0.0, 1.0], # units: [T]
    "field_options": {"max_substeps": 10},
}

## PROBLEMS ##

testem3_composite = {
    "geometry_file": "testem3-composite.gdml",
    "primary_options": {
        "position": [-22, 0, 0],
        "direction": [1, 0, 0],
        "_units": "cgs",
    },
}

_tilecal_angle = 76 * (2 * math.pi / 360)
tilecal = {
    "geometry_file": "atlas-tilecal.gdml",
    "primary_options": {
        "position": [229.801, 0, 0],
        "direction": [math.sin(_tilecal_angle), 0, math.cos(_tilecal_angle)],
    },
}

hgcal = {
    "geometry_file": "cms-hgcal.gdml",
    "primary_options": {
        "position": [0, 0, -899.999],
        "direction": [0, 0, 1],
    },
}

full_cms_run2 = {
    "geometry_file": "cms2018.gdml",
    "cuda_stack_size": 8192,
}

full_cms_run3 = {
    "geometry_file": "cms-run3.gdml",
    "cuda_stack_size": 8192,
}
full_cms_run4 = {
    "geometry_file": "cms-hllhc.gdml",
    "cuda_stack_size": 8192,
}

# List of list of setting dictionaries
problems = [
    [testem3_composite, use_field],
    [tilecal],
    [hgcal],
    [full_cms_run3, use_field],
    [full_cms_run4, use_field],
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


def merge_inputs(problem_dicts):
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

    return inp


def run(app, track_order='none'):
    num_streams = 16
    num_threads = [2**i for i in range(16, 23)]
    if app == 'celer-g4':
        # celer-g4 tracks are per-stream
        num_threads = [n // num_streams for n in num_threads]

    # Check that executable exists
    exe = build_dir / 'bin' / app
    if not shutil.which(exe, os.X_OK):
        msg = f'Unable to locate executable {exe}.'
        raise IOError(msg)

    inputs = [merge_inputs([base_input] + p) for p in problems]

    for inp in inputs:
        # Set number of streams
        if app == 'celer-sim' and not inp['merge_events']:
            os.environ['OMP_NUM_THREADS'] = str(num_streams)
        else:
            os.environ['G4FORCENUMBEROFTHREADS'] = str(num_streams)

        if ('cms' in inp['geometry_file']):
            os.environ["CUDA_HEAP_SIZE"] = "10000000"
            os.environ["CUDA_STACK_SIZE"] = "32000"

        # Set number of events
        inp['primary_options']['num_events'] = num_streams

        # Set track order
        inp['track_order'] = track_order

        geo_name = Path(inp['geometry_file']).stem
        ext = '-' + track_order.replace('_', '-')
        outdir = results_dir / f'scaling{ext}' / app / geo_name
        outdir.mkdir(exist_ok=True, parents=True)

        for i in range(len(num_threads)):
            inits_per_track = 128 if num_threads[i] < 2**19 else 64
            inp['num_track_slots'] = num_threads[i]
            inp['initializer_capacity'] = inits_per_track * num_threads[i]
            outfile = outdir / f'{i}.json'

            # Run app and redirect output to JSON
            print(f'Running {app} on {geo_name} with track_order={track_order} and {num_threads[i]} GPU threads')
            arg_list = [exe, '-']
            proc = subprocess.run(
                arg_list,
                input=json.dumps(inp),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            print(proc.stderr)
            try:
                with open(str(outfile), 'w') as f:
                    json.dump(json.loads(proc.stdout), f)

                with open(str(outfile)) as f:
                    out = json.load(f)
                    if app == 'celer-sim':
                        time = out['result']['runner']['time']['total']
                    else:
                        time = out['result']['time']['total']
                    print(f'GPU threads: {num_threads[i]}, time: {time}')
            except:
                "Error loading json output"


def main():
    app = 'celer-g4'
    for to in ['none', 'init_charge', 'reindex_along_step_action']:
        run(app, to)


if __name__ == '__main__':
    main()
