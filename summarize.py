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

def get_msc(d):
    try:
        msc_model = d['geant_options']['msc']
    except KeyError:
        return d['enable_msc']
    else:
        return msc_model != "none"

def get_num_events_and_primaries(d):
    try:
        primary_gen = d['primary_gen_options']
    except KeyError:
        num_events = d.get('_num_events', None)
        num_primaries = d.get('_num_primaries', None)
    else:
       num_events = primary_gen['num_events']
       num_primaries = num_events * primary_gen['primaries_per_event']
    return (num_events, num_primaries)

def summarize_result(out):
    """Calculate statistics about the tracking behaviors.

    These are basically equivalent to those in StepperTestBase.
    """
    inp, result, internal = (out.get(k) for k in ['input', 'result', 'internal'])
    summary = {}
    if inp is not None:
        (num_events, num_primaries) = get_num_events_and_primaries(inp)
        summary["num_events"] = num_events
        summary["num_primaries"] = num_primaries
    else:
        num_events = 0
        num_primaries = 0

    if result is None:
        return summary

    active = result['active']
    time = result['time']
    steps = sum(active)
    emptying_step = calc_emptying_step(active)
    summary.update({
        "unconverged": result['alive'][-1] + result['initializers'][-1],
        "num_step_iters": len(active),
        "num_steps": steps,
        "emptying_step": emptying_step,
        "setup_time": time['setup'],
        "total_time": time['total'],
        "active_hwm": calc_hwm(result['active']),
        "queue_hwm": calc_hwm(result['initializers']),
        "pre_emptying_time": time['steps'][(emptying_step or 0) - 1],
        "avg_steps_per_primary": steps / num_primaries,
        "avg_time_per_step": time['total'] / steps,
        "avg_time_per_primary": time['total'] / num_primaries,
        "slot_occupancy": steps / (len(active) * inp['max_num_tracks'])
    })

    try:
        summary["action_times"] = time['actions']
    except KeyError:
        # Backward compatibility
        if internal is not None:
            summary["action_times"] = get_action_times(internal['actions'])

    return summary

def summarize_input(inp):
    return {
        'geometry_filename': PurePath(inp['geometry_filename']).name,
        'mag_field': inp['mag_field'] if any(inp['mag_field']) else None,
        'use_device': inp['use_device'],
        'enable_msc': get_msc(inp),
        'max_num_tracks': inp['max_num_tracks'],
    }

def summarize_system(sys):
    try:
        kernels = sys['kernels']
    except KeyError:
        occupancy = None
    else:
        occupancy = {v['name']: v['occupancy'] for v in sys['kernels']}

    get_config = sys['build']['config'].get
    return  {
        'debug': get_config('CELERITAS_DEBUG'),
        'version': sys['build']['version'],
        'geant4': get_config('Geant4_VERSION'),
        'vecgeom': get_config('VecGeom_VERSION'),
        'occupancy': occupancy,
    }

def inp_to_nametuple(d):
    geo_split = PurePath(d['geometry_filename']).name.split('.')
    name = geo_split[0]
    if d.get('mag_field') and any(d['mag_field']):
        name += '+field'
    if get_msc(d):
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
        result = {
            'failure': failure,
            'exception': exception_to_dict(e, context='result')
        }
    else:
        if failure:
            result['failure'] = failure

    return result

def equivalent_systems(a, b):
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        if a[k] == b[k]:
            continue
        if k == 'vecgeom' and (a[k] is None or b[k] is None):
            # Ignore difference between VG and no-VG builds
            continue
        return False
    return True

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
        if not equivalent_systems(systems[0], s):
            consistent = False
            print("WARNING: inconsistent system settings:", s)

    return {
        'input': inp,
        'system': systems[0] if consistent else systems,
        'result': summaries
    }

output_re = re.compile(r'^\d+.json$')

def filter_outputs(paths):
    for p in paths:
        if output_re.match(p.name):
            yield p

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
        result_files = sorted(filter_outputs(outdir.glob("*.json")))
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
