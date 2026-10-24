"""
Figures for the report and the demo. Every figure uses the same colours:
Q-learning = blue, SARSA = orange, random agent = grey.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Rectangle

from warehouse_env import MOVES

# Colour palette (checked for colour-blind readers). Text always stays black or grey.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
AGENT_COLORS = {"Q-learning": "#2a78d6", "SARSA": "#eb6834", "Random": MUTED}
OUTCOME_COLORS = {"delivered": "#0ca30c", "collision": "#d03b3b", "timeout": "#fab219"}
FLOOR = "#ebe9e3"
RACK = "#5f5e5a"
LANE = "#d03b3b"
PARCEL = "#eda100"
LANDMARKS = {"S": ("#d6e5f8", "dock"), "P": ("#fbe3a6", "pickup"), "D": ("#cfeccf", "packing")}
VALUE_CMAP = LinearSegmentedColormap.from_list(
    "value", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
WINDOW = 50  # episodes per moving average
LEARNERS = ("Q-learning", "SARSA")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def figure(width, height, title, subtitle=None):
    """A figure with a left-aligned title block. Returns (fig, top) - axes go below `top`."""
    fig = plt.figure(figsize=(width, height))
    fig.text(0.01, 1 - 0.14 / height, title, ha="left", va="top",
             fontsize=13.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.46 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    return fig, 1 - (1.05 if subtitle else 0.65) / height


# ---------------------------------------------------------------- warehouse map

def _ink_for(rgb):
    """Black text on light colours, white text on dark ones."""
    r, g, b = rgb[:3]
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "white"


def draw_map(ax, env, values=None, norm=None, captions=True):
    """Draw the floor plan. `values` (rows x cols, NaN = no value) turns it into a heat map."""
    def same(r, c, ch):
        return 0 <= r < env.n_rows and 0 <= c < env.n_cols and env.layout[r][c] == ch

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            ch = env.layout[r][c]
            # Floor tiles keep a small gap; racks and the lane join up into solid blocks.
            x0, x1, y0, y1 = c - 0.46, c + 0.46, r - 0.46, r + 0.46
            if ch in "#~":
                color = RACK if ch == "#" else LANE
                x0 -= 0.04 * same(r, c - 1, ch)
                x1 += 0.04 * same(r, c + 1, ch)
                y0 -= 0.04 * same(r - 1, c, ch)
                y1 += 0.04 * same(r + 1, c, ch)
            elif values is not None:
                color = VALUE_CMAP(norm(values[r, c])) if np.isfinite(values[r, c]) else FLOOR
            else:
                color = LANDMARKS[ch][0] if ch in LANDMARKS else FLOOR
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                                   edgecolor="none", zorder=0))
    labels = {}
    for ch, (r, c) in (("S", env.start), ("P", env.pickup), ("D", env.dropoff)):
        if values is None:
            labels[ch] = ax.text(c, r - (0.07 if captions else 0), ch, ha="center", va="center",
                                 fontsize=13, fontweight="bold", color=INK, zorder=3)
            if captions:
                ax.text(c, r + 0.3, LANDMARKS[ch][1], ha="center", va="center",
                        fontsize=6.5, color=INK_2, zorder=3)
        else:  # heat map: small letter in the corner so the arrow stays readable
            fill = VALUE_CMAP(norm(values[r, c])) if np.isfinite(values[r, c]) else (0.92, 0.91, 0.89)
            labels[ch] = ax.text(c - 0.38, r - 0.36, ch, ha="left", va="top", fontsize=7.5,
                                 fontweight="bold", color=_ink_for(fill), zorder=3)
    lane_rows = sorted({r for r, _ in env.hazards})
    if lane_rows:
        row = lane_rows[len(lane_rows) // 2]
        cols = [c for r, c in env.hazards if r == row]
        ax.text(np.mean(cols), row, "FORKLIFT LANE", ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=3)
    ax.set_xlim(-0.5, env.n_cols - 0.5)
    ax.set_ylim(env.n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return labels


def draw_route(ax, result, color, offset=0.0, lw=2.4, arrows=True, zorder=4):
    """Draw a recorded route through the cell centres.

    The leg before the pickup is drawn lighter than the leg carrying the parcel, and the
    two legs are nudged apart so a route that doubles back stays readable.
    """
    states = result["states"]
    xs = np.array([s[1] for s in states], dtype=float)
    ys = np.array([s[0] for s in states], dtype=float)
    carrying = np.array([s[2] for s in states])
    k = int(np.argmax(carrying)) if carrying.any() else len(states) - 1
    legs = [(slice(0, k + 1), -0.11 + offset, 0.45, lw * 0.85)]
    if carrying.any():
        legs.append((slice(k, None), 0.11 + offset, 1.0, lw))
    for part, shift, alpha, width in legs:
        x, y = xs[part] + shift, ys[part] + shift
        ax.plot(x, y, color=color, lw=width, alpha=alpha, zorder=zorder)
        if arrows:
            for i in range(1, len(x), 3):
                dx, dy = x[i] - x[i - 1], y[i] - y[i - 1]
                if dx or dy:
                    mx, my = (x[i] + x[i - 1]) / 2, (y[i] + y[i - 1]) / 2
                    ax.annotate("", xy=(mx + 0.12 * dx, my + 0.12 * dy),
                                xytext=(mx - 0.12 * dx, my - 0.12 * dy), zorder=zorder + 1,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28",
                                                color=color, alpha=alpha, lw=0,
                                                shrinkA=0, shrinkB=0))
    if result["outcome"] != "delivered":  # mark where the robot ended up
        r, c, _ = states[-1]
        ax.add_patch(Circle((c, r), 0.2, facecolor=color, edgecolor=SURFACE, lw=2, zorder=zorder + 2))


def loop_start(states, max_period=4, repeats=5):
    """Step from which the robot only repeats the same few cells until the episode ends, or None."""
    n = len(states)
    start = None
    for p in range(1, max_period + 1):
        i = n - 1 - p
        while i >= 0 and states[i] == states[i + p]:
            i -= 1
        if n - (i + 1) >= p * repeats and (start is None or i + 1 < start):
            start = i + 1
    return start


def describe(result, shortest=None):
    """One-line summary of a recorded episode, e.g. 'delivered in 30 steps'."""
    steps, outcome = result["steps"], result["outcome"]
    if outcome == "delivered":
        best = " (shortest possible)" if shortest and steps == shortest else ""
        return f"delivered in {steps} steps{best}"
    if outcome == "collision":
        return f"drove into the forklift lane after {steps} steps"
    picked = any(s[2] for s in result["states"])
    if loop_start(result["states"]) is not None:
        return "got the parcel, then stuck in a loop" if picked else "stuck going back and forth"
    return "got the parcel, then wandered until time ran out" if picked else \
        f"wandered aimlessly for {steps} steps"


def moves_beside_lane(env, result):
    """How many moves end right next to the forklift lane (not counting P and D themselves)."""
    count = 0
    for r, c, _ in result["states"][1:]:
        if (r, c) in (env.pickup, env.dropoff):
            continue
        if any((r + dr, c + dc) in env.hazards for dr, dc in MOVES):
            count += 1
    return count


# ---------------------------------------------------------------- learning curves

def _moving_average(x, w=WINDOW):
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, x)


def _band(ax, x, runs, color, label=None):
    """Mean line with a +-1 std band across training runs."""
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, mean, color=color, lw=2, label=label)
    return mean


def _reference(ax, y, text, color=INK_2, va="bottom", side="right"):
    """Thin dashed threshold line with its label at one end."""
    ax.axhline(y, color=color, lw=1, ls=(0, (4, 3)), zorder=1)
    lo, hi = ax.get_xlim()
    if side == "right":
        ax.text(hi, y, text + " ", ha="right", va=va, fontsize=8.5, color=color)
    else:
        ax.text(lo, y, " " + text, ha="left", va=va, fontsize=8.5, color=color)


def _end_labels(ax, x_end, values):
    """Direct labels just right of each line's last point."""
    for name, y in values.items():
        ax.annotate(f"{name}  {y:.0f}", (x_end, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)


def _epsilon_floor(ax, history, y):
    """Mark the episode where exploration (epsilon) bottoms out."""
    eps = history["Q-learning"]["epsilon"][0]
    hit = int(np.argmax(eps <= eps.min() + 1e-12))
    ax.axvline(hit, color=AXIS, lw=1, zorder=1)
    ax.text(hit + 10, y, f"from here on only\n{eps.min():.0%} of moves are random", va="center",
            fontsize=8.5, color=MUTED)


def plot_learning_curve(history, summary, path):
    n = history["Q-learning"]["reward"].shape[1]
    x = np.arange(WINDOW, n + 1)
    best = summary["best_possible_return"]
    random_return = summary["agents"]["Random"]["test_avg_return"]["mean"]
    fig, top = figure(12, 4.9, "Learning curve: total reward per training episode",
                      f"{WINDOW}-episode moving average over {summary['config']['runs']} training "
                      "runs (band = ±1 std). The agents still explore while training, so some moves "
                      "are random.")
    gs = fig.add_gridspec(1, 2, top=top, bottom=0.12, left=0.07, right=0.93, wspace=0.16,
                          width_ratios=[1.35, 1])
    ax, zoom = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for name in LEARNERS:
        _band(ax, x, _moving_average(history[name]["reward"]), AGENT_COLORS[name], name)
    ax.set_xlim(0, n)
    ax.set_ylim(min(random_return, ax.get_ylim()[0]) - 40, best + 60)
    _reference(ax, best, f"best possible: {best}", side="left")
    _reference(ax, random_return, f"random agent: {random_return:.0f}", color=MUTED)
    _epsilon_floor(ax, history, y=(random_return + best) / 2)
    ax.set_title("Whole training run")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Return (total reward)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))

    last = 400
    finals = {}
    for name in LEARNERS:
        mean = _band(zoom, x[-(last - WINDOW + 1):],
                     _moving_average(history[name]["reward"][:, -last:]), AGENT_COLORS[name])
        finals[name] = mean[-1]
    zoom.set_xlim(n - last + WINDOW - 1, n)
    zoom.set_ylim(top=best + 8)
    _reference(zoom, best, f"best possible: {best}")
    _end_labels(zoom, n, finals)
    zoom.set_title(f"Last {last} episodes, zoomed in")
    zoom.set_xlabel("Training episode")
    fig.savefig(path)
    plt.close(fig)


def plot_steps(history, summary, path):
    n = history["Q-learning"]["steps"].shape[1]
    x = np.arange(WINDOW, n + 1)
    shortest = summary["shortest_route_steps"]
    fig, top = figure(12, 4.6, "Steps per episode: the robot finds shorter routes",
                      "Left: training episodes (still exploring). Right: test runs of the learned "
                      "route with exploration switched off. 200 steps = the time limit.")
    gs = fig.add_gridspec(1, 2, top=top, bottom=0.13, left=0.07, right=0.93, wspace=0.16)
    ax, test = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)
    for name in LEARNERS:
        _band(ax, x, _moving_average(history[name]["steps"]), AGENT_COLORS[name], name)
        h = history[name]
        test.plot(h["test_episode"], h["test_steps"].mean(axis=0), color=AGENT_COLORS[name],
                  lw=2, label=name)
    for a, title in ((ax, f"Training ({WINDOW}-episode moving average)"),
                     (test, f"Test runs (mean of {summary['config']['runs']} runs)")):
        a.set_xlim(0, n)
        a.set_ylim(0, 215)
        _reference(a, shortest, f"shortest possible: {shortest}", va="top")
        a.set_title(title)
        a.set_xlabel("Training episode")
    ax.set_ylabel("Steps per episode")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_success(history, summary, path):
    n = history["Q-learning"]["delivered"].shape[1]
    x = np.arange(WINDOW, n + 1)
    fig, top = figure(12, 4.6, "Success rate: how often the parcel gets delivered",
                      "Left: training episodes, where exploration still causes some collisions. "
                      "Right: test runs of the learned route with exploration switched off.")
    gs = fig.add_gridspec(1, 2, top=top, bottom=0.13, left=0.07, right=0.93, wspace=0.16)
    ax, test = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for name in LEARNERS:
        h = history[name]
        _band(ax, x, 100 * _moving_average(h["delivered"]), AGENT_COLORS[name], name)
        test.plot(h["test_episode"], 100 * h["test_delivered"].mean(axis=0),
                  color=AGENT_COLORS[name], lw=2, label=name)
    for a, title in ((ax, f"Training ({WINDOW}-episode moving average)"),
                     (test, f"Test runs (mean of {summary['config']['runs']} runs)")):
        a.set_xlim(0, n)
        a.set_ylim(-3, 105)
        a.set_title(title)
        a.set_xlabel("Training episode")
        a.legend(loc="lower right")
    ax.set_ylabel("Episodes delivered (%)")
    fig.savefig(path)
    plt.close(fig)


def plot_outcomes(history, summary, path, bin_size=25):
    n = history["Q-learning"]["delivered"].shape[1] // bin_size * bin_size  # whole blocks only
    edges = np.arange(0, n + 1, bin_size)
    fig, top = figure(12, 4.4, "How training episodes ended",
                      f"Share of episodes in each block of {bin_size}, averaged over "
                      f"{summary['config']['runs']} runs.")
    gs = fig.add_gridspec(1, 2, top=top, bottom=0.14, left=0.07, right=0.98, wspace=0.12)
    for i, name in enumerate(LEARNERS):
        ax = fig.add_subplot(gs[i])
        h = history[name]
        delivered = h["delivered"][:, :n].reshape(len(h["delivered"]), -1, bin_size).mean(axis=(0, 2))
        collision = h["collision"][:, :n].reshape(len(h["collision"]), -1, bin_size).mean(axis=(0, 2))
        timeout = 1 - delivered - collision
        centres = edges[:-1] + bin_size / 2
        ax.stackplot(centres, 100 * delivered, 100 * collision, 100 * timeout,
                     colors=[OUTCOME_COLORS[k] for k in ("delivered", "collision", "timeout")],
                     labels=["delivered the parcel", "drove into the forklift lane", "ran out of time"],
                     edgecolor=SURFACE, linewidth=1.2)
        ax.set_xlim(centres[0], centres[-1])
        ax.set_ylim(0, 100)
        ax.grid(False)
        ax.set_title(name)
        ax.set_xlabel("Training episode")
        if i == 0:
            ax.set_ylabel("Share of episodes (%)")
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def plot_evaluation(summary, path):
    names = ["Q-learning", "SARSA", "Random"]
    metrics = [("test_success_pct", "Delivery success (%)", "{:.0f}%"),
               ("test_avg_steps", "Average steps per episode", "{:.1f}"),
               ("test_avg_return", "Average return per episode", "{:.0f}")]
    fig, top = figure(12, 3.4, "Final test: trained agents vs. a random agent",
                      f"Exploration and learning switched off. {summary['config']['test_episodes']} "
                      f"test episodes per trained agent, mean of {summary['config']['runs']} runs.")
    gs = fig.add_gridspec(1, 3, top=top, bottom=0.08, left=0.09, right=0.97, wspace=0.38)
    for i, (key, title, fmt) in enumerate(metrics):
        ax = fig.add_subplot(gs[i])
        values = [summary["agents"][n][key]["mean"] for n in names]
        y = np.arange(len(names))
        ax.barh(y, values, height=0.42, color=[AGENT_COLORS[n] for n in names])
        span = max(values) - min(0, min(values))
        for yi, v in zip(y, values):
            ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), yi, fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(y, names if i == 0 else [""] * len(names))
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        lo = min(0, min(values))
        ax.set_xlim(lo - (0.25 * span if lo < 0 else 0), max(values) + 0.25 * span)
        if key == "test_success_pct":
            ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- policy and routes

def plot_policy(env, q, name, path):
    """Heat map of how good each cell is, plus the action the agent would take there."""
    values = q.max(axis=3)
    shown = np.zeros(values.shape, dtype=bool)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            for carrying in (0, 1):
                free = (r, c) not in env.walls and (r, c) not in env.hazards
                done = (r, c) == (env.pickup if carrying == 0 else env.dropoff)
                shown[r, c, carrying] = free and not done and np.any(q[r, c, carrying] != 0)
    norm = Normalize(values[shown].min(), values[shown].max())
    fig, top = figure(12, 6.4, f"What the {name} agent learned",
                      "Colour = state value V(s) = max over actions of Q(s, a): darker means more "
                      "future reward expected. Arrow = the action it picks there. Grey = never visited.")
    gs = fig.add_gridspec(2, 2, top=top, bottom=0.04, left=0.02, right=0.98, wspace=0.06,
                          height_ratios=[1, 0.04], hspace=0.12)
    for carrying, title in ((0, "Before the pickup: heading to the parcel"),
                            (1, "Carrying the parcel: heading to the packing station")):
        ax = fig.add_subplot(gs[0, carrying])
        v = np.where(shown[:, :, carrying], values[:, :, carrying], np.nan)
        draw_map(ax, env, values=v, norm=norm)
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                if not shown[r, c, carrying]:
                    continue
                dr, dc = MOVES[int(np.argmax(q[r, c, carrying]))]
                color = _ink_for(VALUE_CMAP(norm(v[r, c])))
                ax.annotate("", xy=(c + 0.27 * dc, r + 0.27 * dr), xytext=(c - 0.27 * dc, r - 0.27 * dr),
                            arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25",
                                            color=color, lw=1.6, shrinkA=0, shrinkB=0), zorder=4)
        ax.set_title(title, fontsize=10.5)
    cax = fig.add_subplot(gs[1, :])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=VALUE_CMAP), cax=cax,
                       orientation="horizontal")
    bar.outline.set_visible(False)
    bar.set_label("State value V(s)", color=INK_2)
    fig.savefig(path)
    plt.close(fig)


def plot_routes(env, routes, summary, path):
    shortest = summary["shortest_route_steps"]
    fig, top = figure(12, 6.0, "Learned routes: Q-learning vs. SARSA",
                      "Lighter line = driving to the parcel, darker line = carrying it. "
                      "Same map, same rewards, same settings: only the update rule differs.")
    gs = fig.add_gridspec(1, 2, top=top, bottom=0.02, left=0.02, right=0.98, wspace=0.06)
    for i, name in enumerate(LEARNERS):
        ax = fig.add_subplot(gs[i])
        draw_map(ax, env)
        draw_route(ax, routes[name], AGENT_COLORS[name])
        beside = moves_beside_lane(env, routes[name])
        ax.set_title(f"{name}: {describe(routes[name], shortest)}", fontsize=11)
        ax.text(-0.45, env.n_rows - 0.35,
                f"{beside} of its {routes[name]['steps']} moves end right beside the forklift lane",
                fontsize=9.5, color=INK_2, va="top")
    fig.savefig(path)
    plt.close(fig)


def plot_learning_progress(env, progress, summary, path):
    """The greedy route at every saved snapshot during training (small multiples)."""
    shortest = summary["shortest_route_steps"]
    episodes = sorted(progress)
    cols = 4
    rows = int(np.ceil(len(episodes) / cols))
    fig, top = figure(13, 1.0 + rows * 3.2, "Learning progress: the Q-learning robot's route as it trains",
                      "Each panel replays the robot's current best guess (exploration off) after the "
                      "given number of training episodes.")
    gs = fig.add_gridspec(rows, cols, top=top, bottom=0.02, left=0.02, right=0.98, wspace=0.08,
                          hspace=0.32)
    for i, ep in enumerate(episodes):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        draw_map(ax, env, captions=False)
        result = progress[ep]
        draw_route(ax, result, AGENT_COLORS["Q-learning"], lw=1.8,
                   arrows=result["outcome"] == "delivered")
        label = "Before training" if ep == 0 else f"After {ep} episodes"
        ax.set_title(label, fontsize=10.5)
        ax.text(-0.45, env.n_rows - 0.35, describe(result, shortest), fontsize=9, color=INK_2, va="top")
    fig.savefig(path)
    plt.close(fig)


def make_all(env, history, summary, checkpoints, routes, progress, out):
    apply_style()
    final = max(checkpoints["Q-learning"])
    plot_learning_curve(history, summary, out / "learning_curve.png")
    plot_steps(history, summary, out / "steps_per_episode.png")
    plot_success(history, summary, out / "success_rate.png")
    plot_outcomes(history, summary, out / "outcomes.png")
    plot_evaluation(summary, out / "evaluation_summary.png")
    for name in LEARNERS:
        plot_policy(env, checkpoints[name][final], name,
                    out / f"policy_map_{name.lower().replace('-', '_')}.png")
    plot_routes(env, routes, summary, out / "learned_routes.png")
    plot_learning_progress(env, progress, summary, out / "learning_progress.png")
