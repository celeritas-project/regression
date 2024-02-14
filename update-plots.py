#!/usr/bin/env python3
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Regenerate plots and tables.
"""

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
import itertools
from collections import namedtuple, defaultdict
from pathlib import Path

with open('plots/style.json') as f:
    mpl.rcParams.update(json.load(f))

import analyze

# NOTE: these are the *used* values. Summit and frontier reserve a core for
# system processes.
system_color = {
    "summit": "#7A954F",
    "frontier": "#BC5544",
    "perlmutter": "#3E92C7",
}

# archgeo_colors = {k: np.array(v, dtype=float) / 255 for k, v in {
#     "cuda/vecgeom": (191, 40, 96),
#     "cuda/vecgeom.spill": (107, 76, 88),
#     "cuda/orange": (153, 168, 50),
#     "cuda/orange.spill": (106, 112, 67),
#     "hip/orange": (57, 140, 173),
#     "hip/orange.spill": (78, 101, 110),
# }.items()}

archgeo_labels = {
    "cuda/vecgeom": "NVIDIA A100 (VecGeom)",
    "cuda/orange": "NVIDIA A100 (ORANGE)",
    "hip/orange": "AMD MI250 (ORANGE)",
}
archgeo_markers = {
    "cuda/vecgeom": ".",
    "cuda/orange": "+",
    "hip/orange": "x",
}

JOULE_PER_WH = 3600



def calc_cpu_gpu_speedup(analysis):
    speedup = analyze.get_cpugpu_ratio(
        analysis.summed["total_time"]
    ).dropna(how="all", axis=0)
    return speedup


def get_where_arch(df, arch):
    slc = df.index.get_level_values("arch") == arch
    return df[slc]


def calc_events_per_task_sec(analysis, like_other=None):
    if like_other is None:
        like_other = analysis.summed
    summed = analysis.summed
    if like_other is not analysis.summed:
        summed = summed.reindex_like(like_other.summed)
    return summed['avg_event_per_time']


def plot_timing(analysis):
    (fig, [run_ax, setup_ax]) = plt.subplots(
        nrows=2,
        gridspec_kw=dict(height_ratios=[3, 1]),
        subplot_kw=dict(yscale="log"),
        layout="constrained"
    )

    analysis.plot_results(run_ax, analysis.summed["total_time"])
    run_ax.grid()
    run_ax.legend()
    run_ax.set_ylabel("Run [s]")
    run_ax.tick_params(labelbottom=False)
    analyze.annotate_metadata(run_ax, analysis)

    analysis.plot_results(setup_ax, analysis.summed["setup_time"])
    setup_ax.grid()
    setup_ax.set_ylabel("Setup [s]")

    return fig


def plot_speedup(analysis, speedup):
    sys = analysis.system
    fig, ax = plt.subplots(layout="constrained")
    analysis.plot_results(ax, speedup)
    ax.grid(which='both')
    num_cpu = analyze.CPU_PER_TASK[analysis.system]
    ax.set_ylabel(f"Speedup ({num_cpu}-CPU / 1-GPU wall time)")
    ax.set_ylim([0, None])

    if analyze.CPU_POWER_PER_TASK[sys] is not None:
        efficiency_factor = (analyze.CPU_POWER_PER_TASK[sys] / analyze.GPU_POWER_PER_TASK[sys])
        oax = ax.twinx()
        red = (.7, .1, .1)
        oax.axhline(1.0, linestyle='--',
                    color=(red + (0.25,)))
        oax.spines['right'].set_color(red)
        oax.set_ylim([0, ax.get_ylim()[1] * efficiency_factor])
        oax.set_ylabel("Power efficiency", color=red)
        for lab in oax.get_yticklabels():
            lab.set_color(red)

    analyze.annotate_metadata(ax, analysis)
    return fig


def plot_steps_vs_primaries(analysis):
    fig, axes = plt.subplots(
        nrows=2, figsize=(4,4), subplot_kw=dict(yscale="log"),
        layout="constrained"
    )
    for (ax, q) in zip(axes, ["step", "primary"]):
        analysis.plot_results(
            ax,
            analyze.inverse_summary(analysis.summed["avg_time_per_" + q])
        )
        ax.set_ylabel(q + " per sec")
        if ax != axes[-1]:
            ax.tick_params(labelbottom=False)
        ax.grid()
        ax.legend()
    analyze.annotate_metadata(axes[0], analysis)
    return fig


def plot_accum_per_step(data, p):
    (fig, axes) = plt.subplots(nrows=2, figsize=(3, 4), sharex=True, layout="constrained")
    for i, ax, plot in zip(itertools.count(),
                           axes,
                           [analyze.plot_counts, analyze.plot_accum_time_inv]):
        objs = plot(ax, data)
        analyze.annotate_metadata(ax, data["_metadata"])
        if i == 0:
            ax.set_xlabel(None)
    return fig


def plot_diff_per_step(data, p):
    (fig, ax) = plt.subplots(figsize=(4, 3), layout="constrained")
    analyze.plot_time_per_step(ax, data, scale=2)
    analyze.annotate_metadata(ax, data["_metadata"])
    return fig


def plot_geo_throughput(analysis, geo_frac):
    (fig, (time_ax, geo_ax)) = plt.subplots(
        nrows=2,
        gridspec_kw=dict(height_ratios=[3, 1]),
        layout="constrained"
    )
    event_rate = analyze.calc_event_rate(analysis)
    time_ax.set_yscale('log')
    p = analysis.plot_results(time_ax, event_rate)
    grid = time_ax.grid()
    time_ax.set_ylabel(r"Throughput [event/s]")
    time_ax.set_ylim([0.5 * event_rate['mean'].min(), None])
    analyze.annotate_metadata(time_ax, analysis)
    time_ax.tick_params(labelbottom=False)
    time_ax.legend()

    analysis.plot_results(geo_ax, geo_frac * 100)
    geo_ax.set_ylabel("Geometry [%]")
    geo_ax.set_ylim([0, 100])
    geo_ax.grid()
    return fig


def dump_event_power(f, analysis):
    rate = analyze.calc_event_rate(analysis)
    power = analysis.power / JOULE_PER_WH # W-h/sec
    rate.loc[:, 'mean'] /= power
    rate.loc[:, 'std'] = power
    return analyze.dump_rate(f, analysis, rate, "[1/W-h]")


def dump_event_rate(f, analysis):
    return analyze.dump_rate(f, analysis, analyze.calc_event_rate(analysis),
                             "[1/s]")


def plot_minimal(system):
    results_dir = Path("results") / system
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path("plots") / system
    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze.Analysis(results_dir)
    print("")
    print(analysis)
    print("="* len(str(analysis)))

    # Check that everything is converged
    unconv = analyze.summarize_instances(analysis.result["unconverged"])["mean"]
    assert not np.any(unconv > 0)

    failures = analysis.failures()
    if failures is not None:
        ftab = analyze.make_failure_table(failures)
    else:
        ftab = None
    failed_file = results_dir / "failed.md"
    if ftab is not None:
        print(f"WARNING: failures occurred: see {failed_file!s}")
        with open(results_dir / "failed.md", "w") as f:
            analyze.dump_markdown(
                f,
                ["Instance", "Failure"],
                np.array([ftab.index, ftab.to_numpy()]).T,
                alignment="<<",
            )
    else:
        failed_file.unlink(missing_ok=True)

    with open(results_dir / "throughput.md", "w") as f:
        dump_event_rate(f, analysis)

    with open(results_dir / "speedup.md", "w") as f:
        analyze.dump_speedup(f, analysis)

    with open(results_dir / "power.md", "w") as f:
        dump_event_power(f, analysis)

    speedup = calc_cpu_gpu_speedup(analysis)

    event_rate = analyze.calc_event_rate(analysis)
    testem3 = event_rate["mean"].xs("testem3-flat+field+msc", level="problem").unstack("arch")
    try:
        del testem3["gpu+sync"]
    except KeyError:
        pass
    print("Speedup for testem3:")
    print(str(testem3 / testem3.loc[("orange", "cpu")]))

    _desc = (speedup["mean"].dropna()).describe()
    print("Speedups: {min:.0f}×–{max:.0f}×".format(**_desc))
    _desc = (speedup["mean"].dropna() * 7).describe()
    print("CPU:GPU equivalence: {min:.0f}×–{max:.0f}×".format(**_desc))

    ### SPEEDUPS ###
    fig = plot_speedup(analysis, speedup)
    fig.savefig(plots_dir / "speedup.pdf", transparent=True)
    fig.savefig(plots_dir / "speedup.png", transparent=False, dpi=150)
    plt.close()

    fig = plot_geo_throughput(analysis, analyze.calc_geo_frac(analysis))
    fig.savefig(plots_dir / "throughput-geo.pdf", transparent=True)
    plt.close()

    return analysis

def plot_all(system):
    analysis = plot_minimal(system)

    results_dir = Path("results") / system
    plots_dir = Path("plots") / system

    # Check that everything is converged
    unconv = analyze.summarize_instances(analysis.result["unconverged"])["mean"]
    assert not np.any(unconv > 0)

    ### TIMING ###
    fig = plot_timing(analysis)
    fig.savefig(plots_dir / "timing.pdf", transparent=True)
    plt.close()

    ### STEPS VS PRIMARIES ###

    fig = plot_steps_vs_primaries(analysis)
    fig.savefig(plots_dir / "steps-vs-primaries.pdf")
    plt.close()

    ### GEO FRACTIONS ###

    geo_frac = analyze.calc_geo_frac(analysis)
    gf_table = geo_frac["mean"].unstack(["geo", "arch"]).applymap(
        analyze.float_fmt_transform(2)
    )
    with open(results_dir / "geo-frac.md", "w") as f:
        analyze.dump_markdown(
            f,
            ["Problem"] + ["/".join(c) for c in gf_table.columns],
            np.concatenate([np.array([gf_table.index]).T, gf_table], axis=1),
            alignment="<" + ">"*gf_table.shape[1]
        )

    ### Action fraction pie charts ###

    mean_action_times = analysis.action_times().xs("mean", axis=1, level=1).T

    for (prob, geo) in itertools.product(
            ["testem15+field", "testem3-flat+field+msc", "cms2018+field+msc"],
            ["vecgeom", "orange"]):
        try:
            plot_times = mean_action_times.xs(
                (prob, geo), axis=1, level=("problem", "geo")
            ).dropna(axis=1, how="all")
        except KeyError:
            plot_times = pd.DataFrame()
        if plot_times.empty:
            print("Missing problem/geo:", prob, geo)
            continue
        md = {k: getattr(analysis, k) for k in ["version", "system"]}
        pieplot = analyze.PiePlotter(plot_times)

        # Loop over CPU/GPU
        for arch in pieplot.arch:
            (fig, ax) = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"),
                                     layout="constrained")
            pieplot(ax, arch)
            name = (prob, geo, arch)
            slashname = "/".join(name)
            fig.text(
                0.98, 0.1, f"{slashname}\n{md['version']} on {md['system']}",
                va="bottom", ha="right",
                fontstyle="italic", color=(0.75,)*4, size="xx-small",
            )

            dashname = "-".join(name)
            fig.savefig(plots_dir / f"actions-{dashname}.pdf", transparent=True)
            plt.close()

    ### Per-step timing ###

    for p in ["cms2018", "cms2018+field+msc"]:
        data = analysis.load_results((p, "vecgeom", "gpu"), 0)
        fig = plot_accum_per_step(data, p)
        fig.savefig(plots_dir / f"accum-per-step-{p}.pdf", transparent=True)
        fig = plot_diff_per_step(data, p)
        fig.savefig(plots_dir / f"time-per-step-{p}.pdf", transparent=True)
        plt.close()

    ### SORTING RESULTS ###

    if 'geant4' not in analysis.summed.index.levels[1]:
        print("skipping vs-geant4 plots because it's not in the results:",
              analysis.summed.index.levels[1])
        return analysis

    throughput = analyze.summarize_instances(analysis.result['avg_event_per_time'])
    throughput = throughput[~analysis.failed_pga]

    (fig, ax) = plt.subplots()
    analysis.plot_sorting(ax, throughput)
    ax.legend()
    plt.title("Sorting speedup using sort_along_step_action")
    ax.set_ylabel("Througput speedup [sorted/unsorted]")
    grid = ax.grid(which='both')
    analyze.annotate_metadata(ax, analysis)
    fig.savefig(plots_dir / "sorting.pdf", transparent=True)
    fig.savefig(plots_dir / "sorting.png", dpi=150)
    plt.close()

    ### THROUGHPUT VS GEANT4 ###

    # remove sorted problems as the don't don't have CPU results
    throughput = throughput.loc[list(set([x for x in throughput.index.get_level_values('problem') if "sort" not in x]))]

    (fig, ax) = plt.subplots(subplot_kw=dict(yscale='log'))
    analysis.plot_results(ax, throughput)
    ax.legend()
    ax.set_ylabel("Throughput [event/s]")
    grid = ax.grid(which='both')
    analyze.annotate_metadata(ax, analysis)
    fig.savefig(plots_dir / "throughput-with-geant.pdf", transparent=True)
    fig.savefig(plots_dir / "throughput-with-geant.png", dpi=150)
    plt.close()

    ref = throughput.xs(('geant4', 'g4'), level=('geo', 'arch'))
    compare = throughput[throughput.index.get_level_values('arch') != 'g4']

    expanded_ref = ref.loc[list(compare.index.get_level_values('problem'))]
    expanded_ref.index = compare.index
    speedup = analyze.calc_summary_ratio(compare, expanded_ref)
    speedup['mean'].unstack('arch')

    (fig, ax) = plt.subplots(subplot_kw=dict(yscale='log'))
    analysis.plot_results(ax, speedup)
    ax.legend()
    ax.set_ylabel("Speedup [C/G4]")
    grid = ax.grid(which='both')
    hline = ax.axhline(1.0, linestyle='-', linewidth=2,
                color=(.7, .1, .1, 0.5,));
    analyze.annotate_metadata(ax, analysis)
    fig.savefig(plots_dir / "speedup-wrt-geant.png", dpi=150)#transparent=True)
    fig.savefig(plots_dir / "speedup-wrt-geant.pdf", transparent=True)
    plt.close()

    return analysis


def plot_per_node(plot_like, analyses, rates):
    (fig, ax) = plt.subplots(layout="constrained", subplot_kw=dict(yscale="log"))
    for k in analyses:
        r = rates[k]
        for arch in ['cpu', 'gpu', 'g4']:
            # events per task-sec
            v = r[r.index.get_level_values("arch") == arch].copy()
            v *= analyze.TASK_PER_NODE[k]
            scat = plot_like.plot_results(ax, v)
            for s in scat:
                s.set_color(system_color[k])
                s.set_label(f"{k.title()} ({arch.upper()})")
    ax.legend(loc='lower left')
    ax.set_xlabel("Problem")
    ax.set_ylabel("Throughput per node [event/s]")
    analyze.annotate_metadata(ax, plot_like)
    grid = ax.grid(which='both')
    return fig


def plot_power(plot_like, analyses, rates):
    (fig, ax) = plt.subplots(layout="constrained")
    for k in analyses:
        r = rates[k]
        for arch in ['cpu', 'gpu', 'g4']:
            v = get_where_arch(r, arch) # events/(task * s)
            power = get_where_arch(analyses[k].power, arch) / JOULE_PER_WH # W-h/sec
            v.loc[:, 'mean'] /= power
            v.loc[:, 'std'] = power
            scat = plot_like.plot_results(ax, v)
            for s in scat:
                s.set_color(system_color[k])
                s.set_label(f"{k.title()} ({arch.upper()})")

    ax.legend()
    ax.set_xlabel("Problem")
    ax.set_ylabel("Efficiency [event/W-h]")
    analyze.annotate_metadata(ax, plot_like)
    grid = ax.grid(which='both')
    return fig


# def plot_kernel_mem(ax, ksdf, colors, labels):
#     labels = ["local_mem", "register_mem"]
#     y = np.arange(len(labels))
#     width = .9 / len(multimem)
#     ynudge = np.linspace(-0.34, 0.34, len(multimem))
#
#     for (i, (k, mem)) in enumerate(multimem.items()):
#         values = np.array(list(mem.values()), dtype=dtype)
#
#         ax.barh(y + ynudge[i], values["register"], width,
#                 color=colors[k], label=f"{pretty_labels[k]}")
#         ax.barh(y + ynudge[i], values["local"], width, left=values["register"],
#                 color=colors[k + ".spill"])#, label=f"Local spill ({pretty_labels[k]})")
#
#     ax.invert_yaxis();
#     ax.set_xlabel("Memory [B]")
#     ax.set_yticks(y, labels)
#     leg = ax.legend()
#     leg.set_title("Register usage (light)\nLocal spill (dark)")
#     leg.get_title().set_fontsize("x-small")
#     return fig


def plot_reg_vs_spill(ksdf):
    (fig, ax) = plt.subplots(layout="constrained")
    for key, ks in ksdf.unstack("name").iterrows():
        k = "/".join(key)
        ks = ks.unstack(level=0)
        s = ax.scatter(ks["register_mem"], ks["local_mem"],
                   c=ks["kernel_index"],
                   marker=archgeo_markers[k], label=archgeo_labels[k])
    ax.set_xlabel("Register usage [B]")
    ax.set_ylabel("Memory spill [B]")
    ax.legend()
    cb = fig.colorbar(s)
    return fig


def plot_occupancy_vs_mem(ksdf):
    (fig, ax) = plt.subplots(layout="constrained")
    for key, ks in ksdf.unstack("name").iterrows():
        k = "/".join(key)
        ks = ks.unstack(level=0)
        tot_mem = ks["register_mem"] + ks["local_mem"]
        s = ax.scatter(ks["occupancy"], tot_mem,
                   c=ks["kernel_index"],
                   marker=archgeo_markers[k], label=archgeo_labels[k])
    #ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Register + spill [B]")
    ax.legend()
    cb = fig.colorbar(s)
    return fig


def plot_occupancy_vs_spill(ksdf):
    (fig, ax) = plt.subplots(layout="constrained")
    for key, ks in ksdf.unstack("name").iterrows():
        k = "/".join(key)
        ks = ks.unstack(level=0)
        s = ax.scatter(ks["occupancy"], ks["local_mem"],
                   c=ks["kernel_index"],
                   marker=archgeo_markers[k], label=archgeo_labels[k])
    #ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Local memory spill [B]")
    ax.legend()
    cb = fig.colorbar(s)
    return fig


def plot_kernels(cuda, hip, problem):
    plots_dir = Path("plots")

    kernel_stats = {
        "cuda/vecgeom": analyze.load_kernels(cuda, problem, "vecgeom"),
        "cuda/orange": analyze.load_kernels(cuda, problem, "orange"),
        "hip/orange": analyze.load_kernels(hip, problem, "orange"),
    }
    ksdf = analyze.kernel_stats_dataframe(kernel_stats)

    with open("kernel-occupancy.md", "w") as f:
        analyze.dump_markdown(
            f,
            list(ksdf.index.names) + ["local", "register", "occupancy"],
            np.concatenate([
                np.array([list(v) for v in ksdf.index]).T,
                [
                    ksdf["local_mem"].apply("{:d}".format),
                    ksdf["register_mem"].apply("{:d}".format),
                    ksdf["occupancy"].apply("{:.03f}".format)
                ],
            ], axis=0).T,
            alignment="<<<>>>"
        )

    ## REGISTERS VS SPILL ##
    fig = plot_reg_vs_spill(ksdf)
    #fig.savefig(plots_dir / "reg-vs-spill.png")
    fig.savefig(plots_dir / "reg-vs-spill.pdf", transparent=True)
    plt.close()

    ## OCCUPANCY VS MEM ##
    fig = plot_occupancy_vs_mem(ksdf)
    #fig.savefig(plots_dir / "occupancy-vs-mem.png")
    fig.savefig(plots_dir / "occupancy-vs-mem.pdf", transparent=True)
    plt.close()

    ## OCCUPANCY VS SPILL ##
    fig = plot_occupancy_vs_spill(ksdf)
    #fig.savefig(plots_dir / "occupancy-vs-spill.png")
    fig.savefig(plots_dir / "occupancy-vs-spill.pdf", transparent=True)
    plt.close()

    print("Large memory kernels:")
    print(ksdf[ksdf["local_mem"] > 64])
    return ksdf

def main():
    analyses = {}

    # Plot individual results
    analyses["summit"] = plot_all("summit")
    analyses["crusher"] = plot_minimal("crusher")
    analyses["frontier"] = plot_minimal("frontier")
    analyses["perlmutter"] = plot_like = plot_all("perlmutter")

    del analyses["crusher"]

    # Compare multiple systems
    plots_dir = Path("plots")

    rates = {k: calc_events_per_task_sec(v, plot_like)
             for (k, v) in analyses.items()}

    fig = plot_per_node(plot_like, analyses, rates)
    fig.savefig(plots_dir / "event-per-node.pdf", transparent=True)
    fig.savefig(plots_dir / "event-per-node.png", transparent=False, dpi=150)
    plt.close()

    fig = plot_power(plot_like, analyses, rates)
    fig.savefig(plots_dir / "event-per-energy.pdf", transparent=True)
    fig.savefig(plots_dir / "event-per-energy.png", transparent=False, dpi=150)

    # Plot kernels
    plot_kernels(analyses["perlmutter"], analyses["frontier"], "testem3-flat+field+msc")

if __name__ == '__main__':
    main()
