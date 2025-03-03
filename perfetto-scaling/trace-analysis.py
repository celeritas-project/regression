from perfetto.trace_processor import TraceProcessor
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

with open("../plots/style.json") as f:
    mpl.rcParams.update(json.load(f))

tp = TraceProcessor(trace="celer-plugin.perfetto-trace")

qr__neutral_it = tp.query(
    "SELECT  ts, dur, name FROM slice WHERE name='along-step-neutral' ORDER BY ts ASC"
)
qr_uniform_it = tp.query(
    "SELECT  ts, dur, name FROM slice WHERE name='along-step-general-linear' ORDER BY ts ASC"
)
qr_active_counter_it = tp.query(
    "SELECT ts, value FROM counter WHERE track_id=3 ORDER BY ts ASC"
)

along_step_dur = []
active_count = []
ts = []
time_per_track = []
start_time = next(iter(qr__neutral_it)).ts
for neutral, uniform, active in zip(
    qr__neutral_it, qr_uniform_it, qr_active_counter_it
):
    along_step_dur.append((uniform.dur + neutral.dur) / 1e6)
    active_count.append(active.value)
    ts.append((neutral.ts - start_time) / 1e9)
    time_per_track.append(along_step_dur[-1] / active.value)

# Identify where time per track starts to increase
# Find the minimum point after some initial data
start_idx = 400  # Skip first few points which might be noisy
min_idx = start_idx + time_per_track[start_idx:].index(min(time_per_track[start_idx:]))
threshold_time = ts[min_idx]
threshold_active = active_count[min_idx]

fig, (ax1, ax3) = plt.subplots(
    2,
    1,
    sharex=True,
    gridspec_kw={"height_ratios": [4, 1]},
    subplot_kw=dict(yscale="log")
)
ax2 = ax1.twinx()
ax2.set_yscale('log')
ax1.set_ylabel("Propagation time [ms]")
ax2.set_ylabel("Active tracks")

# Loop over tuples
for data, color, axis, side in [
    (along_step_dur, "tab:red", ax1, "left"),
    (active_count, "tab:blue", ax2, "right"),
]:
    axis.plot(ts, data, color=color)
    axis.spines[side].set_color(color)
    axis.yaxis.label.set_color(color)
    axis.tick_params(axis="y", colors=color)

ax1.grid()
ax2.axhline(y=threshold_active, color=color, linestyle=':', alpha=0.7)

# Bottom subplot for time per track
color = "tab:green"
ax3.set_xlabel("Wall time [s]")
ax3.set_ylabel("Per track [ms]")
ax3.plot(
    ts,
    time_per_track,
    color=color,
)
ax3.tick_params(axis="y")
ax3.grid()


# Add vertical lines to both subplots
ax1.axvline(x=threshold_time, color="black", linestyle="--", alpha=0.7)
ax3.axvline(x=threshold_time, color="black", linestyle="--", alpha=0.7)

# Add annotation
ax2.annotate(
    "Time per track\nstarts increasing",
    xy=(threshold_time, threshold_active),
    xytext=(threshold_time - 10, 0.25 * threshold_active),
    arrowprops=dict(facecolor="black", arrowstyle='-|>'),
    horizontalalignment="right",
    va="top",
)

#plt.title("Propagation time and active tracks")
# Save the figure with high resolution
plt.savefig("propagation_time_vs_active_tracks.pdf")
plt.show()
