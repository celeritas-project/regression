from perfetto.trace_processor import TraceProcessor
import matplotlib.pyplot as plt

tp = TraceProcessor(trace='celer-plugin.perfetto-trace')

qr__neutral_it = tp.query("SELECT  ts, dur, name FROM slice WHERE name='along-step-neutral' ORDER BY ts ASC")
qr_uniform_it = tp.query("SELECT  ts, dur, name FROM slice WHERE name='along-step-general-linear' ORDER BY ts ASC")
qr_alive_counter_it = tp.query("SELECT ts, value FROM counter WHERE track_id=3 ORDER BY ts ASC")

along_step_dur = []
alive_count = []
ts = []
time_per_track = []
start_time = iter(qr__neutral_it).__next__().ts
for neutral, uniform, alive in zip(qr__neutral_it, qr_uniform_it, qr_alive_counter_it):
    along_step_dur.append((uniform.dur + neutral.dur) / 1e6)
    alive_count.append(alive.value)
    ts.append((neutral.ts - start_time) / 1e9)
    time_per_track.append(along_step_dur[-1] / alive.value)

plt.rcParams.update({
    'font.size': 14,  # Increase font size
})
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

color = 'tab:red'
ax1.set_ylabel('propagation time [ms]', fontsize=16)
ax1.plot(ts, along_step_dur, color=color, linewidth=3)
ax1.tick_params(axis='y',  color=color)
ax1.set_yscale('log')

ax2 = ax1.twinx()
ax2.spines['left'].set_color(color)
ax2.spines['left'].set_linewidth(1.5)

color = 'tab:blue'
ax2.spines['right'].set_color(color)
ax2.spines['right'].set_linewidth(1.5)
ax2.set_ylabel('active tracks', fontsize=16)
ax2.plot(ts, alive_count, color=color, linewidth=3)
ax2.tick_params(axis='y', color=color)
ax2.axhline(y=3000, color=color, linestyle=':', linewidth=2, alpha=0.7)
ax2.set_yscale('log')

# Bottom subplot for time per track
color = 'tab:green'
ax3.set_xlabel('wall time [s]', fontsize=16)
ax3.set_ylabel('ms per track', fontsize=16)
ax3.plot(ts, time_per_track, color=color, linewidth=3)
ax3.tick_params(axis='y')
ax3.set_yscale('log')

# Identify where time per track starts to increase
# Find the minimum point after some initial data
start_idx = 400  # Skip first few points which might be noisy
min_idx = start_idx + time_per_track[start_idx:].index(min(time_per_track[start_idx:]))
threshold_time = ts[min_idx]

# Add vertical lines to both subplots
ax1.axvline(x=threshold_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(x=threshold_time, color='black', linestyle='--', linewidth=2, alpha=0.7)

# Add annotation
ax2.annotate('Time per track\nstarts increasing', 
             xy=(threshold_time, ax2.get_ylim()[1]*0.003),
             xytext=(threshold_time-8, ax2.get_ylim()[1]*0.0005),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=14, fontweight='bold',
             horizontalalignment='right')

plt.title('Propagation time and alive tracks', fontsize=20)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# Save the figure with high resolution
plt.savefig('propagation_time_vs_alive_tracks.pdf', dpi=300, bbox_inches='tight')
plt.show()
