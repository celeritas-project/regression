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

cpu_per_gpu = {
    "wildstyle": 32,
    "summit": 7,
    "frontier": 7,
    "crusher": 7,
    "perlmutter": 16,
}

cpu_power = {
    "summit": None, # not specified with just raw chips
    "frontier": 280 / 6, # 64-core AMD “Optimized 3rd Gen EPYC”
    "perlmutter": 280 / 4, # AMD EPYC 7763
}

gpu_power = {
    "summit": 250, # V100
    "frontier": 500 / 2, # MI250x
    "perlmutter": 250, # A100
}

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
    "cuda/vecgeom": "NVIDIA V100 (VecGeom)",
    "cuda/orange": "NVIDIA V100 (ORANGE)",
    "hip/orange": "AMD MI250 (ORANGE)",
}
archgeo_markers = {
    "cuda/vecgeom": ".",
    "cuda/orange": "+",
    "hip/orange": "x",
}

def plot_timing(analysis):
    (fig, [run_ax, setup_ax]) = plt.subplots(
        nrows=2,
        gridspec_kw=dict(height_ratios=[3, 1]),
        subplot_kw=dict(yscale="log")
    )

    analysis.plot_results(run_ax, analysis.summed["total_time"])
    run_ax.legend();
    run_ax.set_ylabel("Run [s]")
    run_ax.tick_params(labelbottom=False)
    analyze.annotate_metadata(run_ax, analysis)

    analysis.plot_results(setup_ax, analysis.summed["setup_time"])
    setup_ax.set_ylabel("Setup [s]")

    fig.tight_layout()
    return fig


def plot_speedup(analysis, speedup):
    sys = analysis.system
    fig, ax = plt.subplots(layout="constrained")
    analysis.plot_results(ax, speedup)
    num_cpu = cpu_per_gpu[analysis.system]
    ax.set_ylabel(f"Speedup ({num_cpu}-CPU / 1-GPU wall time)")
    ax.set_ylim([0, None])

    if cpu_power[sys] is not None:
        efficiency_factor = (cpu_power[sys] / gpu_power[sys])
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
        nrows=2, figsize=(4,4), subplot_kw=dict(yscale="log")
    )
    for (ax, q) in zip(axes, ["step", "primary"]):
        analysis.plot_results(
            ax,
            analyze.inverse_summary(analysis.summed["avg_time_per_" + q])
        )
        ax.set_ylabel(q + " per sec")
        if ax != axes[-1]:
            ax.tick_params(labelbottom=False)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_accum_per_step(data, p):
    (fig, axes) = plt.subplots(nrows=2, figsize=(3, 4), sharex=True)
    for i, ax, plot in zip(itertools.count(),
                           axes,
                           [analyze.plot_counts, analyze.plot_accum_time_inv]):
        objs = plot(ax, data)
        analyze.annotate_metadata(ax, data["_metadata"])
        if i == 0:
            ax.set_xlabel(None)
    fig.tight_layout()
    return fig


def plot_diff_per_step(data, p):
    (fig, ax) = plt.subplots(figsize=(4, 3))
    analyze.plot_time_per_step(ax, data, scale=2)
    analyze.annotate_metadata(ax, data["_metadata"])
    fig.tight_layout()
    return fig


def plot_geo_throughput(analysis, geo_frac):
    (fig, (time_ax, geo_ax)) = plt.subplots(
        nrows=2,
        gridspec_kw=dict(height_ratios=[3, 1])
    )
    analyze.plot_event_rate(time_ax, analysis)
    time_ax.tick_params(labelbottom=False)
    time_ax.legend()

    analysis.plot_results(geo_ax, geo_frac * 100)
    geo_ax.set_ylabel("Geometry [%]")
    geo_ax.set_ylim([0, 100])
    fig.tight_layout()
    return fig


def plot_all(system):
    results_dir = Path("results") / system
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path("plots") / system
    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze.Analysis(results_dir)
    print(analysis)

    # Check that everything is converged
    unconv = analyze.summarize_instances(analysis.result["unconverged"])["mean"]
    assert not np.any(unconv > 0)


    with open(results_dir / "throughput.md", "w") as f:
        analyze.dump_event_rate(f, analysis)

    with open(results_dir / "speedup.md", "w") as f:
        analyze.dump_speedup(f, analysis)

    summed = analysis.summed
    times = summed[("total_time", "mean")].unstack()
    speedup = analyze.get_cpugpu_ratio(summed["total_time"]).dropna(how="all", axis=0)

    event_rate = analyze.calc_event_rate(analysis)
    testem3 = event_rate["mean"].xs("testem3-flat+field+msc", level="problem").unstack("arch")
    print(str(testem3 / testem3.loc[("vecgeom", "cpu")]))

    _desc = (speedup["mean"].dropna()).describe()
    print("Speedups: {min:.0f}×–{max:.0f}×".format(**_desc))
    _desc = (speedup["mean"].dropna() * 7).describe()
    print("CPU:GPU equivalence: {min:.0f}×–{max:.0f}×".format(**_desc))

    ### TIMING ###
    fig = plot_timing(analysis)
    fig.savefig(plots_dir / "timing.pdf", transparent=True)
    plt.close()


    ### SPEEDUPS ###
    fig = plot_speedup(analysis, speedup)
    fig.savefig(plots_dir / "speedups.pdf", transparent=True)
    fig.savefig(plots_dir / "speedup.png", transparent=False, dpi=150)
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

    ### GEO THROUGHPUT ###

    fig = plot_geo_throughput(analysis, geo_frac)
    fig.savefig(plots_dir / "throughput-geo.pdf", transparent=True)
    plt.close()

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

    return analysis


def plot_minimal(system):
    results_dir = Path("results") / system
    analysis = analyze.Analysis(results_dir)
    print(analysis)

    # Check that everything is converged
    unconv = analyze.summarize_instances(analysis.result["unconverged"])["mean"]
    assert not np.any(unconv > 0)

    with open(results_dir / "throughput.md", "w") as f:
        analyze.dump_event_rate(f, analysis)

    with open(results_dir / "speedup.md", "w") as f:
        analyze.dump_speedup(f, analysis)

    plots_dir = Path("plots") / system
    return analysis


def plot_compare(old, new):
    new_rates = analyze.calc_event_rate(new)
    old_rates = analyze.calc_event_rate(old, old.summed.loc[new_rates.index])

    # TODO: add relative.md
    # rel = (new_rates["mean"] / old_rates["mean"]).unstack()

    problems = old.problems()
    problem_to_abbr = old.problem_to_abbr(problems)
    p_to_i = dict(zip(problems, itertools.count()))

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    for (offset, system, rates) in [
        (-0.05, old.system, old_rates),
        (0.05, new.system, new_rates)
    ]:
        for arch in ["cpu", "gpu"]:
            summary = rates.xs(arch, level="arch")
            index = np.array([
                p_to_i[p] for p in summary.index.get_level_values("problem")
            ], dtype=float)
            index += offset

            mark = analyze.ARCH_SHAPES[arch]
            ax.errorbar(index, summary["mean"], summary["std"],
                        capsize=0, fmt="none", ecolor=(0.2,)*3)
            label = "{system} ({count} {arch})".format(
                system=system.title(),
                count=cpu_per_gpu[system] if arch == "cpu" else 1,
                arch=arch.upper()
            )
            scat = ax.scatter(index, summary["mean"],
                              c=system_color[system], marker=mark,
                              label=label)

    xax = ax.get_xaxis()
    xax.set_ticks(np.arange(len(problems)))
    xax.set_ticklabels(list(problem_to_abbr.values()), rotation=90)
    grid = ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_ylabel(r"Event rate [1/s]")
    analyze.annotate_metadata(ax, new)
    plt.tight_layout()
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
    (fig, ax) = plt.subplots()
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
    fig.tight_layout()
    return fig


def plot_occupancy_vs_mem(ksdf):
    (fig, ax) = plt.subplots()
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
    fig.tight_layout()
    return fig


def plot_occupancy_vs_spill(ksdf):
    (fig, ax) = plt.subplots()
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
    fig.tight_layout()
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


# Plot individual results
summit = plot_all("summit")
crusher = plot_minimal("crusher")
frontier = plot_minimal("frontier")
perlmutter = plot_all("perlmutter")

# Compare
fig = plot_compare(summit, frontier)
fig.savefig("plots/frontier-vs-summit.pdf")
plt.close()

# Plot kernels
plot_kernels(summit, frontier, "testem3-flat+field+msc")
