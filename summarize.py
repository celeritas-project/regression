#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
from pathlib import PurePath
import re

def calc_emptying_step(active):
    if not active:
        # No statistics saved
        return None
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
    if not counts:
        return None
    hq, hi = max((q, i) for (i, q) in enumerate(counts))
    return {"index": hi, "count": hq}

def clean_up_result(result):
    """Reduce verbosity/diffs in output"""
    try:
        vols = result["internal"]["geometry"]["volumes"]
    except KeyError:
        # Possibly g4 with no 'internal' result
        return

    labels = []
    for lab in vols["label"]:
        labels.append(lab.partition("@")[0])
    vols["label"] = labels

def get_action_times(actions):
    return {a['label']: a['time'] for a in actions if a.get('time', 0) > 0}

def get_num_events_and_primaries(d):
    primary_gen = d.get('primary_options')
    if primary_gen:
        num_events = primary_gen['num_events']
        num_primaries = num_events * primary_gen['primaries_per_event']
    else:
        num_events = d.get('_num_events', None)
        num_primaries = d.get('_num_primaries', None)
    return (num_events, num_primaries)

def get_num_track_slots(inp):
    num_track_slots = inp['num_track_slots']
    return num_track_slots

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

    if inp['_exe'] == "celer-sim":
        result = result['runner']

    time = result['time']
    total_time = time['total']
    summary["setup_time"] = time['setup']
    summary["total_time"] = total_time
    summary["avg_time_per_primary"] = total_time / num_primaries
    summary["avg_event_per_time"] = num_events / total_time
    summary["gpu_energy_wh"] = result.get("gpu_energy_wh", 0.0)

    if inp['_exe'] == "celer-sim":
        def squeeze(r):
            if len(r) == 1:
                return r[0]
            return r

        def mean(r):
            return sum(r) / len(r)

        def get_stream_counts(key, op):
            r = result[key]
            if not r:
                return None
            return op(r)

        active = get_stream_counts('active', squeeze)
        inits = get_stream_counts('initializers', squeeze)
        alive = get_stream_counts('alive', squeeze)

        try:
            steps = get_stream_counts('num_steps', sum)
            step_iters = get_stream_counts('num_step_iterations', squeeze)
            aborted = get_stream_counts('num_aborted', sum)
            queue_hwm = get_stream_counts('max_queued', mean)
        except (KeyError, IndexError):
            # < 0.4.3
            steps = sum(active) if active else None
            step_iters = len(active) if active else None
            aborted = alive[-1] if alive else None
            queue_hwm = calc_hwm(inits)

        active_hwm = (calc_hwm(active)
                      if active and isinstance(active[0], int) else None)
        emptying_step = (calc_emptying_step(active)
                         if active and isinstance(active[0], int) else None)
        preempty_time = (time['steps'][0][emptying_step - 1]
                         if emptying_step else None)
        slot_oc = (steps / (step_iters * inp['num_track_slots'])
                   if step_iters and isinstance(step_iters, int) else None)

        summary.update({
            "unconverged": aborted,
            "num_step_iters": step_iters,
            "num_steps": steps,
            "emptying_step": emptying_step,
            "warmup_time": time.get('warmup', None),
            "active_hwm": active_hwm,
            "queue_hwm": queue_hwm,
            "avg_steps_per_primary": steps / num_primaries,
            "avg_time_per_step": total_time / steps if steps else None,
            "avg_step_per_time": steps / total_time,
            "slot_occupancy": slot_oc,
            "action_times": time['actions'],
            "pre_emptying_time": preempty_time,
        })


    return summary

def summarize_input(inp):
    field = inp.get('field')
    geo_file = inp.get('geometry_file')

    return {
        'geometry_name': PurePath(geo_file).name,
        'field': field,
        'use_device': inp.get('use_device'),
        'enable_msc': inp['physics_options']['msc'] != "none",
        'num_track_slots': get_num_track_slots(inp),
        'merge_events': inp.get('merge_events'),
    }

def summarize_system(r):
    sys = r['system']
    try:
        kernels = sys['kernels']
    except KeyError:
        occupancy = None
    else:
        occupancy = {v['name']: v['occupancy'] for v in sys['kernels']}

    # Version/configure changed in v0.5.1
    get_config = sys['build']['config'].get
    get_version = (get_config("versions") or {}).get

    # Sizes available with v0.5.2 onward
    try:
        sizes = r['internal']['core-sizes']
    except KeyError:
        # Possibly celer-g4 (no 'internal')
        sizes = None

    return  {
        'debug': get_config('debug') or get_config('CELERITAS_DEBUG'),
        'version': sys['build']['version'],
        'geant4': get_version('Geant4') or get_config('Geant4_VERSION'),
        'vecgeom': get_version('VecGeom') or get_config('VecGeom_VERSION'),
        'occupancy': occupancy,
        'sizes': sizes,
        'openmp': get_config('openmp'),
        'build_type': get_config('build_type'),
    }

def inp_to_nametuple(inp):
    geo_split = PurePath(inp['geometry_file']).name.split('.')
    name = geo_split[0]
    if inp.get('field') and any(inp['field']):
        name += '+field'
    if inp['physics_options']['msc'] != "none":
        name += '+msc'

    geo = inp['_geometry']

    arch = "gpu" if inp.get("use_device", False) else "cpu"
    if inp.get('_exe') == 'celer-g4':
        if not inp['_use_celeritas']:
            arch = "g4"
        else:
            arch = arch + "+g4"
    if arch == "gpu" and (
            inp.get("action_times", False) or inp.get("sync", False)):
        arch += "+sync"

    return (name, geo, arch)

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
            sys_sum = summarize_system(r)
        except KeyError as e:
            print("Couldn't summarize system: missing key", e)
        else:
            systems.append(sys_sum)
            if inp is None:
                inp = summarize_input(r['input'])

    if inp is None:
        return {'result': "No instances ran successfully"}

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
                try:
                    results.append(json.load(f))
                except json.decoder.JSONDecodeError as e:
                    print(f"Failed to read file '{r}': {e!s}")
        summaries[subdir] = summary = summarize_all(results)
        summary['name'] = name

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
