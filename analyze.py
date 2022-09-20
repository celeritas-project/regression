#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import pandas as pd
from pathlib import Path

def unstack_subdict(df):
    result = pd.DataFrame(list(df.values), index=df.index)
    result.columns.name = df.name
    return result


def summarize_instances(df):
    """TODO: probably a better way to do this with a combination of
    stack/unstack or groupby"""
    transformed = {}
    for k in df.columns:
        temp = df[k].unstack().T.describe().T[['count', 'mean', 'std']]
        temp.columns.name = k
        transformed[temp.columns.name] = temp
    result = pd.concat(transformed, axis=1, names=[df.columns.name, "summary"])
    return result


def sum_instance(series):
    return series.groupby(level=['problem', 'geo', 'arch']).sum()


class Analysis:
    def __init__(self, basedir):
        basedir = Path(basedir)
        with open(basedir / 'index.json') as f:
            index = {tuple(name): dirname
                     for (dirname, name) in json.load(f)}
        with open(basedir / 'summaries.json') as f:
            summaries = {tuple(v.pop('name')): v
                         for v in json.load(f).values()}

        input = pd.DataFrame([v['input'] for v in summaries.values()],
                               index=summaries.keys())
        result = pd.DataFrame([v['result'] for v in summaries.values()],
                               index=summaries.keys())
        result = result.stack()
        result.index.names = ['problem', 'geo', 'arch', 'instance']
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

