#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
from pathlib import PurePath
import re

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

def summarize_result(result, internal):
    """Calculate statistics about the tracking behaviors.

    These are basically equivalent to those in StepperTestBase.
    """
    active = result['active']
    time = result['time']
    steps = sum(active)

    emptying_step = calc_emptying_step(active)
    summary = {
        "num_primaries": active[0],
        "unconverged": result['alive'][-1] + result['initializers'][-1],
        "num_step_iters": len(active),
        "num_steps": steps,
        "emptying_step": emptying_step,
        "total_time": time['total'],
        "queue_hwm": calc_queue_hwm(result['initializers']),
        "pre_emptying_time": time['steps'][(emptying_step or 0) - 1],
        "action_times": get_action_times(internal['actions']),
        "avg_steps_per_primary": steps / active[0],
        "avg_time_per_step": time['total'] / steps,
        "avg_time_per_primary": time['total'] / active[0],
    }
    return summary

def summarize_input(inp):
    try:
        msc = inp['enable_msc']
    except KeyError:
        # Newer
        print(inp)
        msc = inp['geant_options']['msc']

    return {
        'geometry_filename': PurePath(inp['geometry_filename']).name,
        'mag_field': inp['mag_field'] if any(inp['mag_field']) else None,
        'use_device': inp['use_device'],
        'enable_msc': msc,
    }

def summarize_system(sys):
    kernels = sys['kernels']
    return  {
        'debug': sys['build']['config']['CELERITAS_DEBUG'],
        'version': sys['build']['version'],
        'occupancy': {v['name']: v['occupancy'] for v in sys['kernels']},
    }

def inp_to_nametuple(d):
    geo_split = PurePath(d['geometry_filename']).name.split('.')
    name = geo_split[0]
    if d.get('mag_field') and any(d['mag_field']):
        name += '+field'
    if d['enable_msc']:
        name += '+msc'

    return (
        name,
        "vecgeom" if geo_split[-1] == 'gdml' else "orange",
        "gpu" if d["use_device"] else "cpu"
    )

failure_re = re.compile('(error|warning|exception|critical)', re.IGNORECASE)

def summarize_failure(out):
    try:
        return out['result']['exception']
    except (KeyError, TypeError):
        result = {}
        for k in ['stderr', 'stdout']:
            if k not in out:
                continue
            result[k] = [line for line in out[k]
                         if failure_re.search(line) is not None]
        return result

def exception_to_dict(e, context=None):
    return {'type': str(type(e)), 'str': str(e), 'context': context}

def summarize_one(out):
    failure = summarize_failure(out)
    if failure:
        return failure

    try:
        result = summarize_result(out['result'], out['internal'])
    except Exception as e:
        return exception_to_dict(e, context='result')

    return result

def summarize_all(instances):
    """Create a summary of all instances that ran.
    """
    instances = list(instances)
    summaries = [summarize_one(result) for result in instances]
    try:
        single = next((result for result in instances
                        if 'system' in result))
    except StopIteration:
        print("Can't summarize: no runs have system output!")
        return {'result': summaries}

    return {
        'input': summarize_input(single['input']),
        'system': summarize_system(single['system']),
        'result': summaries
    }

def main(index_filename):
    import json
    from pathlib import Path
    from pprint import pprint

    # Load index
    index_filename = Path(index_filename)
    with open(index_filename) as f:
        problems = json.load(f)
    results_dir = index_filename.parent

    # Load results
    summaries = {}
    for subdir, name in problems.items():
        outdir = results_dir / subdir
        result_files = sorted(outdir.glob("*.json"))
        results = []
        for r in result_files:
            with open(r) as f:
                results.append(json.load(f))
        summaries[subdir] = summary = summarize_all(results)
        pprint(summary)
        summary['name'] = name

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
