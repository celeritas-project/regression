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
from enum import IntEnum

import pandas as pd
import numpy as np

Islc = pd.IndexSlice

KernelCategory = IntEnum("KernelCategory", ["GEO", "PHYS", "GP"], start=0)

RESULT_LEVELS = ('problem', 'geo', 'arch', 'instance')
GEO_COLORS = {'orange': '#F6A75E', 'vecgeom': '#5785B7'}
ARCH_SHAPES = {'gpu': 'x', 'cpu': 'o'}
KERNEL_CATEGORY_LABELS = ["Geometry", "Physics", "Geo&Phys"]
KERNEL_ORDERING = {
    'along-step-neutral': KernelCategory.GEO,
    'along-step-general-linear': KernelCategory.GP,
    'along-step-uniform-msc': KernelCategory.GP,
    'initialize-tracks': KernelCategory.GP,
    'extend-from-primaries': KernelCategory.GP,
    'extend-from-secondaries': KernelCategory.GP,
    'geo-boundary': KernelCategory.GEO,
    'physics-discrete-select': KernelCategory.PHYS,
    'pre-step': KernelCategory.PHYS,
}

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

def _calc_scale_and_label(arr):
    dtype = type(arr[0])
    scale10 = np.floor(np.log10(np.max(arr)))
    if scale10 == 0:
        return (dtype(1), "")

    return (dtype(10**scale10), " [$\\times 10^{{{n}}}$]".format(n =
                                                                 int(scale10)))


def plot_counts(ax, out):
    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    lines = []
    def plot(ax, *args, **kwargs):
        line, = ax.plot(*args, **kwargs)
        lines.append(line)

    get_counts = CountGetter(out, stream=0)
    active = np.asarray(get_counts('active'), dtype=float)
    alive = np.asarray(get_counts('alive'), dtype=float)
    (norm, scale_label) = _calc_scale_and_label(active)

    plot(ax, active / norm, '-', color=(blue + (0.5,)), label='Active')
    plot(ax, alive / norm, '-', color=blue, label='Alive')
    ax.set_xlabel('Step iteration')
    ax.set_ylabel('Tracks' + scale_label, color=blue)
    ax.set_axisbelow(True)
    ax.grid()

    oax = ax.twinx()
    inits = np.asarray(get_counts('initializers'))
    plot(oax, inits / norm, '--', color=red, label='Queued')
    oax.axhline(out['input']['initializer_capacity'] / norm, linestyle='--',
                color=(red + (0.25,)))
    oax.set_ylabel("Initializers" + scale_label, color=red)

    max_init_idx = np.argmax(inits)
    max_init_val = inits[max_init_idx]
    text = re.sub(r'([-+.0-9]+)e\+?(-)?0*([0-9]+)', r'$\1\\times 10^{\2\3}$',
                  f'{max_init_val:.2g}')
    oax.annotate(text + ' queued', xy=(max_init_idx, max_init_val / norm), xycoords='data',
                 xytext=(30, 10), textcoords='offset points',
                 size='x-small',
                 bbox=dict(boxstyle="round,pad=.2", fc=(0.9, 0.9, 0.9, 0.8) , ec=(0.2,)*3),
                 arrowprops=dict(arrowstyle="->", ec=red, lw=1,
                                 connectionstyle="arc3,rad=0.2"))

    oax.spines['left'].set_color(blue)
    oax.spines['right'].set_color(red)
    oax.set_axisbelow(True)

    legend = ax.legend(lines, [l.get_label() for l in lines])

    return {
        'ax': ax,
        'oax': oax,
        'legend': legend,
    }


def plot_time_per_step(ax, outp):
    get_counts = CountGetter(outp, stream=0)
    get_step_time = StepTimeGetter(outp, stream=0)

    active = np.asarray(get_counts('active'), dtype=float)
    stime = np.asarray(get_step_time()) * 1000
    def _xy(idx):
        return np.array([active[idx], stime[idx]])

    hq, hi = max((q, i) for (i, q) in enumerate(active))
    hq //= 2
    _, filling = min((q, i) for (i, q) in enumerate(active[:hi]) if q > hq)
    _, draining = min((q, i) for (i, q) in enumerate(active[hi:], start=hi) if q > hq)

    alpha = np.ones_like(active, dtype=float)
    alpha[active == outp['input']['num_track_slots']] = .05

    (norm, active_label) = _calc_scale_and_label(active)
    active /= norm

    ax.set_axisbelow(True)
    ax.axhline(0, linestyle='-', lw=0.5, color=(0.75,)*3, zorder=-2)
    ax.axvline(0, linestyle='-', lw=0.5, color=(0.75,)*3, zorder=-2)

    ax.plot(active, stime, marker='', color="0.9", zorder=-1, lw=.5)
    scat = ax.scatter(active, stime, c=np.arange(len(active)), alpha=0.8, s=3,
                      edgecolors='none')
    ax.annotate('All primaries active', xy=_xy(0), xycoords='data',
                xytext=(30, 0), textcoords='offset points',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=.5))
    ax.annotate('Filling', xy=(_xy(filling)), xycoords='data',
                xytext=(_xy(filling) * [1.1, 0.7]), textcoords='data',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=.5))
    ax.annotate('Draining', xy=(_xy(draining)), xycoords='data',
                xytext=(_xy(draining) * [0.7, 1.3]), textcoords='data',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=.5))
    ax.set_xlabel("Number of active tracks" + active_label)
    ax.set_ylabel("Time per step [ms]")

    cb = ax.get_figure().colorbar(scat)
    cb.set_label('Step iteration')

    return {
        'ax': ax,
        'cb': cb,
    }


def plot_accum_time_inv(ax, outp):
    get_counts = CountGetter(outp, 0)
    get_step_time = StepTimeGetter(outp, 0)

    active = np.asarray(get_counts('active'), dtype=float)
    stime = np.asarray(get_step_time())

    accum_time = np.cumsum(stime)
    accum_steps = np.cumsum(active)

    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    ax.plot(np.arange(len(accum_steps)), accum_time, marker='', color=blue)
    ax.set_xlabel("Step iteration")
    ax.set_ylabel("Total wall time [s]", color=blue)
    ax.grid()
    ax.set_axisbelow(True)

    oax = ax.twinx()
    oax.set_axisbelow(True)

    (norm, accum_steps_label) = _calc_scale_and_label(accum_steps)
    accum_steps /= norm

    oax.plot(np.arange(len(accum_steps)), accum_steps, linestyle='-.',
             marker='', color=(red + (0.5,)))
    oax.set_ylabel("Total number of steps" + accum_steps_label, color=red)
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


def float_fmt_transform(digits):
    format = "{{:.{}f}}".format(digits).format
    def transform(val):
        if np.isnan(val):
            return "—"
        return format(val)
    return transform


def get_action_priority(k):
    try:
        return KERNEL_ORDERING[k]
    except KeyError:
        # Physics model
        return KernelCategory.PHYS


def autopct_format(pctvalue):
    if pctvalue < 5:
        return ""
    return "{:1.0f}%".format(pctvalue)


class PiePlotter:
    def __init__(self, times):
        import matplotlib.pyplot as plt
        self._colormaps = plt.colormaps

        self.times = times.dropna()

        actions = list(self.times.index)
        ca = sorted([(get_action_priority(a), a) for a in actions])
        (category, actions) = zip(*ca)

        cmap = self._colormaps["tab20c"] # 5 groups of 4 shades
        def _get_color(color, shade = 0):
            shade = shade % 4
            return cmap((color % 5) * 4 + shade)

        self.outer_colors = _get_color(np.arange(len(KernelCategory)))
        self.actions = np.array(actions)

        cat = np.array([int(c) for c in category])
        self.catbound = np.concatenate([cat[:-1] != cat[1:], [True]])
        self.catlabels = [KERNEL_CATEGORY_LABELS[c] for c in cat[self.catbound]]

    def __call__(self, ax, arch):
        width = 0.3
        angle = 90.0 # degrees
        legend_thresh = 0.02

        series = self.times[arch]
        inner = np.array([series[t] for t in self.actions])
        outer = np.cumsum(inner)[self.catbound]
        outer = np.concatenate([[outer[0]], np.diff(outer)])

        # Plot outer ring (categories
        (wedges, texts, autotexts) = ax.pie(
            outer,
            autopct=autopct_format, pctdistance=0.85,
            radius=1, colors=self.outer_colors,
            wedgeprops=dict(width=width, edgecolor='w'), startangle=angle,
        )
        outer_legend = ax.legend(wedges, self.catlabels,
                  loc="upper right",
                  bbox_to_anchor=(0.5, 0, 0.5, 1))
        ax.add_artist(outer_legend)

        # Determine kernels to higlight
        inner_frac = inner / np.sum(inner)
        inner_cmap = self._colormaps["plasma"]

        slc = inner_frac > legend_thresh
        num_inner = np.count_nonzero(slc)
        ascending_idx = np.argsort(np.argsort(inner_frac[slc]))
        inner_colors = np.zeros((inner.size, 4))
        inner_colors[slc, :] = inner_cmap(
                np.linspace(0.0, 1.0, num_inner)[ascending_idx])
        inner_colors[~slc, :] = [0.5, 0.5, 0.5, 1.0]

        (wedges, texts) = ax.pie(
            inner,
            radius=(1 - width), colors=inner_colors,
            wedgeprops=dict(width=width, edgecolor='w'), startangle=angle,
        )

        # Generate legend with all real kernels and one stand-in gray kernel
        wedges = np.array(wedges)
        actions = self.actions[slc].tolist()
        if not np.all(slc):
            wedges = wedges[slc].tolist() + [wedges[~slc][0]]
            actions += ["Fast kernel (<{:.0f}%)".format(legend_thresh * 100)]

        inner_legend = ax.legend(
            wedges, actions,
            loc="center",
            fontsize='xx-small'
        )
        ax.add_artist(inner_legend)
def main():
    # Generate table from wildstyle failures
    pass
