#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Analysis and plotting utilities for regression data.
"""
import json
import re

from pathlib import Path

import pandas as pd
import numpy as np

Islc = pd.IndexSlice


RESULT_LEVELS = ('problem', 'geo', 'arch', 'instance')


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
    grouped = groupby_notinstance(obj)
    return grouped.describe().loc[Islc[:], Islc[:, ['count', 'mean', 'std']]]


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


class Analysis:
    def __init__(self, basedir):
        basedir = Path(basedir)
        with open(basedir / 'index.json') as f:
            index = {tuple(name): dirname
                     for (dirname, name) in json.load(f)}
        with open(basedir / 'summaries.json') as f:
            summaries = {tuple(v.pop('name')): v
                         for v in json.load(f).values()}

        input = pd.DataFrame([v.get('input', {}) for v in summaries.values()],
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
        self.invalid = ~result['failure'].isna()
        self.version = self._load_version(summaries)

    def _load_version(self, summaries):
        versions = set()
        for s in summaries.values():
            try:
                temp_sys = s["system"]
            except KeyError:
                pass
            else:
                if isinstance(temp_sys, list):
                    versions.update(ts["version"] for ts in temp_sys)
                else:
                    versions.add(temp_sys["version"])
        if len(versions) > 1:
            print("WARNING: multiple versions present in same summary:", versions)
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
        return unstack_subdict(self.result['failure'].dropna())

    def action_times(self):
        result = self.result
        invalid = ~result['failure'].isna()
        return summarize_instances(
            unstack_subdict(result['action_times'][~invalid]))

    def active_hwm(self):
        hwm = self.result['active_hwm']
        return summarize_instances(unstack_subdict(hwm[~invalid]))

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

    def __str__(self):
        return f"Analysis for Celeritas {self.version} on {self.system}"

    @property
    def system(self):
        return self.basedir.name

def plot_counts(ax, out):
    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    lines = []
    def plot(ax, *args, **kwargs):
        line, = ax.plot(*args, **kwargs)
        lines.append(line)

    plot(ax, out['result']['active'], '-', color=(blue + (0.5,)), label='Active')
    plot(ax, out['result']['alive'], '-', color=blue, label='Alive')
    ax.set_xlabel('Step iteration')
    ax.set_ylabel('Tracks', color=blue)

    oax = ax.twinx()
    inits = np.array(out['result']['initializers'])
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
    r = outp['result']
    active = np.asarray(r['active'])
    stime = np.asarray(r['time']['steps'])

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
    r = outp['result']
    active = np.asarray(r['active'])
    stime = np.asarray(r['time']['steps'])

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
        s = f"{md.version} on {md.system}"
    else:
        # Assume data from a single result
        name = "/".join(md['name'])
        s = f"{name}.{md['instance']}\n{md['version']} on {md['system']}"

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

