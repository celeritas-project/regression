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
from pathlib import Path
from pprint import pprint
import re

build_dirs = {
    ("orange", "reldeb"): Path("/Users/seth/.local/src/celeritas/build"),
    ("orange", "opt"): "",
    ("vecgeom", "reldeb"): "",
    ("vecgeom", "opt"): "",
}

testdir = Path(__file__).parent

base_input = {
    "brem_combined": True,
    "enable_diagnostics": False,
    "initializer_capacity": 1e6,
    "mag_field": [0.0, 0.0, 0.0],
    "max_num_tracks": 32,
    "max_steps": 256,
    "secondary_stack_factor": 3.0,
    "sync": True,
    "use_device": False
}

geant_input = {
    "rayleigh": True,
    "eloss_fluctuation": True,
    "brems": "all",
    "lpm": True,
    "msc": "none"
}

simple_cms = {
    "_geometry": "orange",
    "geometry_filename": str(testdir / "input" / "simple-cms.org.json"),
    "hepmc3_filename": str(testdir / "input" / "simple-cms-10k.hepmc3"),
    "physics_filename": str(testdir / "input" / "simple-cms.gdml"),
}

problems = {
    "simple-cms": simple_cms,
}

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
    return {a['label']: a['time'] for a in actions if a['time'] > 0}

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
    }
    return summary

failure_re = re.compile('(error|warning|exception|critical)', re.IGNORECASE)
def summarize_failure(out):
    try:
        return out['result']['exception']
    except KeyError:
        matches = []
        for k in ['stderr', 'stdout']:
            matches.extend(line for line in matches[k]
                           if failure_re.search(line) is not None)
        return matches

async def run_celeritas(problem, instance, debug=True):
    inp = base_input.copy()
    inp.update(problems[problem])

    build = build_dirs[(inp["_geometry"], "reldeb" if debug else "opt")]
    cmd = build / "app/demo-loop"
    print(f"{instance}: awaiting creation of '{cmd}'")
    proc = await asyncio.create_subprocess_exec(
        cmd, "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    # TODO: define alarm to send process SIGINT after too long an interval
    # TODO: monitor output, e.g. https://gist.github.com/kalebo/1e085ee36de45ffded7e5d9f857265d0

    inp["seed"] = 20220904 + instance
    print(f"{instance}: awaiting communcation")
    out, err = await proc.communicate(input=json.dumps(inp).encode())

    print(f"{instance}: complete")
    assert proc.returncode is not None
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
    problem = "simple-cms"
    result = await asyncio.gather(*(run_celeritas(problem, i)
                                    for i in range(4)))
    [summaries, results] = zip(*result)
    pprint(summaries)
    with open('summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)
    with open('results.json', 'w') as f:
        json.dump(summaries, f, indent=0, sort_keys=True)

asyncio.run(main())
