#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Analysis and plotting utilities for regression data.
"""
import itertools
import json
import re

from pathlib import Path, PurePosixPath

import pandas as pd
import numpy as np

Islc = pd.IndexSlice


RESULT_LEVELS = ('problem', 'geo', 'arch', 'instance')
GEO_COLORS = {'orange': '#F6A75E', 'vecgeom': '#5785B7'}
ARCH_SHAPES = {'gpu': 'x', 'cpu': 'o'}

def unstack_subdict(df):
    result = pd.DataFrame(list(df.values), index=df.index)
    result.columns.name = df.name
    return result


def groupby_notinstance(obj):
    """
    Return an object suitable for ``describe``, ``sum``, etc.
    """
    return obj.groupby(level=RESULT_LEVELS[:-1])


def summarize_instances(obj):
    descr = groupby_notinstance(obj).describe()
    if isinstance(obj, pd.Series):
        return descr[['count', 'mean', 'std']]
    return descr.loc[Islc[:], Islc[:, ['count', 'mean', 'std']]]


def inverse_summary(summary):
    result = summary.copy()
    mean = 1/summary['mean']
    rel = summary['std'] / summary['mean']
    std = rel * mean
    return pd.DataFrame({'count': summary['count'], 'mean': mean, 'std': std})


def get_cpugpu_ratio(summary):
    mean = summary['mean'].unstack()
    re = summary['std'].unstack() / mean
    ratio = mean['cpu'] / mean['gpu']
    std = ratio * np.hypot(re['gpu'], re['cpu'])
    return pd.DataFrame({'mean': ratio, 'std' : std})


def calc_event_rate(results, summary):
    ppe = summary[('num_primaries', 'mean')] / summary[('num_events', 'mean')]
    event_rate = inverse_summary(summary['avg_time_per_primary'])
    event_rate['mean'] /= ppe
    event_rate['std'] /= ppe
    return event_rate


class ProblemAbbreviator:
    def __init__(self):
        input_dir = Path(__file__).parent / "input"
        with open(input_dir / "problem-abbr.json") as f:
            self.geo_abbrev = json.load(f)

    def __call__(self, inp):
        geo, *_ = inp['geometry_filename'].partition('.')
        bits = [self.geo_abbrev[geo]]
        if inp.get('mag_field') and any(inp['mag_field']):
            bits.append('F')
        if inp['enable_msc']:
            bits.append('M')

        return "".join(bits)

abbreviate_problem = ProblemAbbreviator()

class Analysis:
    def __init__(self, basedir):
        basedir = Path(basedir)
        with open(basedir / 'index.json') as f:
            index = {tuple(name): dirname
                     for (dirname, name) in json.load(f)}
        with open(basedir / 'summaries.json') as f:
            summaries = {tuple(v.pop('name')): v
                         for v in json.load(f).values()}

        input = pd.DataFrame([v.get('input') or {} for v in summaries.values()],
                             index=summaries.keys())
        input.index.names = RESULT_LEVELS[:-1]

        result = pd.DataFrame([v['result'] for v in summaries.values()],
                               index=summaries.keys())
        result = result.stack()
        result.index.names = RESULT_LEVELS
        result = unstack_subdict(result)

        self.basedir = basedir
        self.index = index
        self.input = input
        self.result = result
        if 'failure' in result:
            self.valid = result['failure'].isna()
        else:
            self.valid = pd.Series(True, index=result.index)
        self.version = self._load_version(summaries)

        failed_probs = (~self.successful).groupby(level='problem').any()
        self.failed_problems = set(failed_probs.index[failed_probs])

    def _load_version(self, summaries):
        versions = set()
        def _lstripv(text):
            if text.startswith("v"):
                return text[1:]
            return text

        for s in summaries.values():
            try:
                temp_sys = s["system"]
            except KeyError:
                pass
            else:
                if isinstance(temp_sys, list):
                    versions.update(_lstripv(ts["version"]) for ts in temp_sys)
                else:
                    versions.add(_lstripv(temp_sys["version"]))
        if len(versions) > 1:
            print("WARNING: multiple versions present in same summary:",
                  versions)
        elif not versions:
            print("WARNING: no version found")
        return " or ".join(versions)

    def load_results(self, name, instance):
        subdir = self.basedir / self.index[name]
        with open(subdir / f"{instance:d}.json") as f:
            result = json.load(f)

        try:
            version = result['system']['build']['version']
        except KeyError:
            version = self.version
        result['_metadata'] = {
            'version': version,
            'system': self.system,
            'name': name,
            'instance': instance,
        }
        return result

    def failures(self):
        try:
            failures = self.result['failure']
        except KeyError:
            return
        return unstack_subdict(failures.dropna())

    def action_times(self):
        result = self.result
        return summarize_instances(
            unstack_subdict(result['action_times'][self.valid]))

    def active_hwm(self):
        hwm = self.result['active_hwm']
        return summarize_instances(unstack_subdict(hwm[self.valid]))

    def problems(self):
        """Get an ordered list of problem names.
        """
        skip = set()
        result = []
        for (p, *_) in self.index:
            if p not in skip:
                result.append(p)
                skip.add(p)
        return result

    def problem_to_abbr(self, problems=None):
        if problems is None:
            problems = self.problems()
        result = {}
        for p in problems:
            inputs = self.input.xs(p, level='problem')
            inputs = inputs.dropna(subset='geometry_filename')
            if len(inputs):
                value = abbreviate_problem(inputs.iloc[0])
                if p in self.failed_problems:
                    value += "*"
            else:
                print(f"WARNING: no inputs available for {p}")
                result[p] = "*"
            result[p] = value
        return result

    def __str__(self):
        return f"Analysis for Celeritas v{self.version} on {self.system}"

    @property
    def system(self):
        return self.basedir.name

    @property
    def invalid(self):
        return ~self.valid

    @property
    def successful(self):
        return self.valid & (self.result['unconverged'] == 0)

    def plot_results(self, ax, df):
        problems = self.problems()
        problem_to_abbr = self.problem_to_abbr(problems)
        p_to_i = dict(zip(problems, itertools.count()))
        get_levels = df.index.get_level_values

        # One data point for each row, with geometries close to each other
        index = np.array([p_to_i[p] for p in get_levels('problem')], dtype=float)
        index += [(0.1 if g == 'orange' else -0.05) for g in get_levels('geo')]
        color = np.array([GEO_COLORS[g] for g in get_levels('geo')])

        if 'arch' in df.index.names:
            slc_mark = [(a.upper(), get_levels('arch') == a, ARCH_SHAPES[a])
                        for a in ['cpu', 'gpu']]
        else:
            slc_mark = [(None, slice(None), 's')]

        result = []
        for lab, slc, mark in slc_mark:
            temp_idx = index[slc]
            temp_sum = df.loc[slc]
            ax.errorbar(temp_idx, temp_sum['mean'], temp_sum['std'],
                        capsize=0, fmt='none', ecolor=(0.2,)*3)
            scat = ax.scatter(temp_idx, temp_sum['mean'], c=color[slc],
                              marker=mark, label=lab)
            result.append(scat)

        xax = ax.get_xaxis()
        xax.set_ticks(np.arange(len(problems)))
        xax.set_ticklabels(list(problem_to_abbr.values()), rotation=90)
        grid = ax.grid()
        ax.set_axisbelow(True)
        return scat
    
def CountGetter(out, stream):
    result = out['result']
    try:
        result = result['runner']
    except KeyError:
        # v0.2 and before #774
        def get_counts(key):
            return result[key][-1]
    else:
        def get_counts(key):
            return result[key][stream]
    
    return get_counts

def StepTimeGetter(out, stream):
    result = out['result']
    try:
        result = result['runner']
    except KeyError:
        # v0.2 and before #774
        time = result['time']
        def get_step_time():
            return time['steps']
    else:
        time = result['time']
        def get_step_time():
            return time['steps'][stream]

    return get_step_time
    
def plot_counts(ax, out):
    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    lines = []
    def plot(ax, *args, **kwargs):
        line, = ax.plot(*args, **kwargs)
        lines.append(line)

    get_counts = CountGetter(out, stream=0)
        
    plot(ax, get_counts('active'), '-', color=(blue + (0.5,)), label='Active')
    plot(ax, get_counts('alive'), '-', color=blue, label='Alive')
    ax.set_xlabel('Step iteration')
    ax.set_ylabel('Tracks', color=blue)

    oax = ax.twinx()
    inits = np.array(get_counts('initializers'))
    plot(oax, inits, '--', color=red, label='Queued')
    oax.axhline(out['input']['initializer_capacity'], linestyle='--', color=(red + (0.25,)))
    oax.set_ylabel('Initializers', color=red)

    max_init_idx = np.argmax(inits)
    max_init_val = inits[max_init_idx]
    text = re.sub(r'([-+.0-9]+)e\+?(-)?0*([0-9]+)', r'$\1\\times 10^{\2\3}$',
                  f'{max_init_val:.2g}')
    oax.annotate(text + ' queued', xy=(max_init_idx, max_init_val), xycoords='data',
                 xytext=(30, 10), textcoords='offset points',
                 size='x-small',
                 bbox=dict(boxstyle="round,pad=.2", fc=(0.9, 0.9, 0.9, 0.8) , ec=(0.2,)*3),
                 arrowprops=dict(arrowstyle="->", ec=red, lw=1,
                                 connectionstyle="arc3,rad=0.2"))

    oax.spines['left'].set_color(blue)
    oax.spines['right'].set_color(red)

    legend = ax.legend(lines, [l.get_label() for l in lines])

    return {
        'ax': ax,
        'oax': oax,
        'legend': legend,
    }


def plot_time_per_step(ax, outp):
    get_counts = CountGetter(outp, stream=0)
    get_step_time = StepTimeGetter(outp, stream=0)

    active = np.asarray(get_stream_counts('active'))
    stime = np.asarray(get_step_time())

    alpha = np.ones_like(active, dtype=float)
    alpha[active == outp['input']['max_num_tracks']] = .05

    # Manually scale
    mega_active = active * 1e-6

    def _xy(idx):
        return np.array([mega_active[idx], stime[idx]])

    ax.plot(mega_active, stime, marker='', color="0.9", zorder=-1, lw=.5)
    scat = ax.scatter(mega_active, stime, c=np.arange(len(mega_active)), s=6) #, alpha=alpha)
    ax.annotate('All primaries active', xy=_xy(0), xycoords='data',
                xytext=(30, 0), textcoords='offset points',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=.5))
    ax.annotate('First secondaries', xy=_xy(2), xycoords='data',
                xytext=(-15, 50), textcoords='offset points',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=.5))
    ax.annotate('Filling', xy=(_xy(3) * 1.1), xycoords='data',
                xytext=(_xy(5) * 1.2), textcoords='data',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="<-", ec=".2", lw=.5))
    ax.annotate('Draining', xy=(_xy(-100) * .8), xycoords='data',
                xytext=(_xy(-80) * .9), textcoords='data',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="<-", ec=".2", lw=.5))
    ax.set_xlabel(r'Number of active tracks [$\times 10^{6}$]')
    ax.set_ylabel('Time per step [s]')
    cb = ax.get_figure().colorbar(scat)
    cb.set_label('Step iteration')

    return {
        'ax': ax,
        'oax': oax,
        'cb': cb,
    }


def plot_accum_time(ax, outp):
    get_counts = CountGetter(outp, 0)
    get_step_time = StepTimeGetter(outp, 0)

    active = np.asarray(get_counts('active'))
    stime = np.asarray(get_step_time())

    accum_time = np.cumsum(stime)
    accum_steps = np.cumsum(active)

    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    ax.plot(accum_time, accum_steps, marker='', color=blue)
    ax.set_ylabel('Total number of steps')
    ax.set_xlabel('Total wall time (s)')
    ax.grid()

    oax = ax.twinx()
    oax.plot(accum_time, np.arange(len(accum_steps)), linestyle='-.', marker='', color=(red + (0.5,)))
    oax.set_ylabel("Number of step iterations", color=red)
    oax.spines['left'].set_color(blue)
    oax.spines['right'].set_color(red)

    return {
        'ax': ax,
        'oax': oax,
    }


def annotate_metadata(obj, md, **kwargs):
    """Draw a little caption on a figure or axis with result metadata.
    """
    if isinstance(md, Analysis):
        s = f"v{md.version} on {md.system}"
    else:
        # Assume data from a single result
        name = "/".join(md['name'])
        s = f"{name}.{md['instance']}\nv{md['version']} on {md['system']}"

    try:
        # Assume obj is axes to get layout coordinates
        transform = obj.transAxes
    except AttributeError:
        # Hope obj is a figure
        transform = None

    text_kwargs = dict(va='bottom', ha='right',
        fontstyle='italic', color=(0.5,)*3, size='xx-small',
        transform=transform,
        zorder=-100
    )
    text_kwargs.update(kwargs)

    return obj.text(0.98, 0.02, s, **text_kwargs)


def make_failure_table(failures):
    if failures is None:
        failures = pd.DataFrame(dtype=object)
    flist = []
    idx = []
    for key, err in failures.iterrows():
        idx.append("{}/{}+{} ({:d})".format(*key))
        tp = err.get("type")
        if tp == "DebugError":
            f = PurePosixPath(err["file"])
            err["file"] = f.name
            text = "{which}: `{condition}` at `{file}:{line}`".format(**err)
        elif tp == "RuntimeError":
            f = PurePosixPath(err["file"])
            err["file"] = f.name
            text = "{which} error: `{what}` at `{file}:{line}`".format(**err)
        elif isinstance(err["stdout"], list) and err["stdout"]:
            text = "`{}`".format(err["stdout"][-1])
        elif isinstance(err["stderr"], list) and err["stderr"]:
            for line in err["stderr"]:
                if line.startswith("celeritas: CUDA error"):
                    text = line
                    break
            else:
                # Use final line
                text = line
            text = "`{}`".format(line)
        else:
            text = "(unknown failure)"
        flist.append(text)
    return pd.Series(flist, index=idx, name="Failure", dtype=object)

def main():
    # Generate table from wildstyle failures
    pass
