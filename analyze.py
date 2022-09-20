#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import pandas as pd
from pathlib import Path

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
        version = None
        for s in summaries.values():
            try:
                test_version = s["system"]["version"]
            except KeyError:
                pass
            else:
                if version is None:
                    version = test_version
                assert version == test_version
        return version

    def load_results(self, name, instance):
        subdir = self.basedir / self.index[name]
        with open(subdir / f"{instance:d}.json") as f:
            return json.load(f)

    def failures(self):
        invalid = self.invalid.unstack()
        invalid = invalid.sum(axis=1) / len(invalid.columns)
        unconverged = (sum_instance(result['unconverged']) /
                       sum_instance(result['num_primaries']))
        return pd.DataFrame({'failed': invalid, 'unconverged': unconverged})

    def action_times(self):
        result = self.result
        invalid = ~result['failure'].isna()
        return summarize_instances(
            unstack_subdict(result['action_times'][~invalid]))

    def active_hwm(self):
        hwm = self.result['active_hwm']
        return summarize_instances(unstack_subdict(hwm[~invalid]))

    def __str__(self):
        return f"Analysis for Celeritas {self.version} on {self.basedir.name}"

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

    ax.legend(lines, [l.get_label() for l in lines])

    return {
        'ax': ax,
        'oax': oax,
    }
