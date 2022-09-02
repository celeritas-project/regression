#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""

- Loop over all problems
- Launch simultaneously on multiple cores (different seed per run!)
- Save overall times from all runs, and output from one run
- Catch failure message and save
"""

build_dirs = {
    ('orange', 'reldeb'): "",
    ('orange', 'opt'): "",
    ('vecgeom', 'reldeb'): "",
    ('vecgeom', 'opt'): "",
}

