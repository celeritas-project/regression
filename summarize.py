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

def calc_hwm(counts):
    hq, hi = max((q, i) for (i, q) in enumerate(counts))
    return {"index": hi, "count": hq}

def get_action_times(actions):
    return {a['label']: a['time'] for a in actions if a.get('time', 0) > 0}

def summarize_result(out):
    """Calculate statistics about the tracking behaviors.

    These are basically equivalent to those in StepperTestBase.
    """
    inp, result, internal = (out[k] for k in ['input', 'result', 'internal'])
    active = result['active']
    time = result['time']
    steps = sum(active)
    try:
        primary_gen = inp['primary_gen_options']
    except KeyError:
        num_events = inp.get('_num_events', None)
        num_primaries = inp.get('_num_primaries', None)
    else:
       num_events = primary_gen['num_events']
       num_primaries = num_events * primary_gen['primaries_per_event']


    emptying_step = calc_emptying_step(active)
    summary = {
        "num_events": num_events,
        "num_primaries": num_primaries,
        "unconverged": result['alive'][-1] + result['initializers'][-1],
        "num_step_iters": len(active),
        "num_steps": steps,
        "emptying_step": emptying_step,
        "total_time": time['total'],
        "active_hwm": calc_hwm(result['active']),
        "queue_hwm": calc_hwm(result['initializers']),
        "pre_emptying_time": time['steps'][(emptying_step or 0) - 1],
        "action_times": get_action_times(internal['actions']),
        "avg_steps_per_primary": steps / active[0],
        "avg_time_per_step": time['total'] / steps,
        "avg_time_per_primary": time['total'] / active[0],
        "slot_occupancy": steps / (len(active) * inp['max_num_tracks'])
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
        'max_num_tracks': inp['max_num_tracks'],
    }

def summarize_system(sys):
    try:
        kernels = sys['kernels']
    except KeyError:
        occupancy = None
    else:
        occupancy = {v['name']: v['occupancy'] for v in sys['kernels']}
    return  {
        'debug': sys['build']['config']['CELERITAS_DEBUG'],
        'version': sys['build']['version'],
        'occupancy': occupancy,
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

failure_re = re.compile(
    r'(error|warning|critical'
    r'|exception|assertion|failed|what\(\)'
    r'|\w+\.(hh|cc):\d+'
    r')', re.IGNORECASE
)

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

    try:
        result = summarize_result(out)
    except Exception as e:
        return {
            'failure': failure,
            'exception': exception_to_dict(e, context='result')
        }

    return result

def summarize_all(instances):
    """Create a summary of all instances that ran.
    """
    instances = list(instances)
    summaries = [summarize_one(result) for result in instances]

    systems = []
    inp = None
    for r in instances:
        try:
            sys_sum = summarize_system(r['system'])
        except KeyError as e:
            print("Couldn't summarize system: missing key", e)
        else:
            systems.append(sys_sum)
            if inp is None:
                inp = summarize_input(r['input'])

    consistent = bool(systems)
    for s in systems[1:]:
        if s != systems[0]:
            consistent = False
            print("WARNING: inconsistent system settings:", s)

    return {
        'input': inp,
        'system': systems[0] if consistent else systems,
        'result': summaries
    }

def main(index_filename):
    import json
    from pathlib import Path

    # Load index
    index_filename = Path(index_filename)
    with open(index_filename) as f:
        problems = json.load(f)
    results_dir = index_filename.parent

    if isinstance(problems, dict):
        # Legacy
        problems = problems.items()

    # Load results
    summaries = {}
    for subdir, name in problems:
        outdir = results_dir / subdir
        print("Processing", outdir)
        result_files = sorted(outdir.glob("*.json"))
        results = []
        for r in result_files:
            with open(r) as f:
                results.append(json.load(f))
        summaries[subdir] = summary = summarize_all(results)
        summary['name'] = name

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
