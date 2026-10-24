"""
Live demo of the warehouse robot. Run train.py first (if you forget, this trains quickly itself).

    python demo.py                   # story: untrained -> partly trained -> fully trained robot
    python demo.py --compare         # Q-learning vs. SARSA side by side: shortest vs. safer route
    python demo.py --live            # watch a brand-new robot learn, learning curve updating live
    python demo.py --ascii           # the story in the terminal, no window needed
    python demo.py --save demo.gif   # save the animation as a GIF instead of opening a window

Add --slip 0.1 to use the models trained on a slippery floor (run train.py --slip 0.1 first).
"""

import argparse
import json
import time
import webbrowser
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

import plots
from agents import QLearningAgent, RandomAgent
from train import (episodes_for, greedy_rollout, load_models, make_agent, make_env, results_dir,
                   run_episode, test_policy)
from warehouse_env import best_possible_return

EVENTS = {"bump": "bumped into a rack  (-5)", "pickup": "picked up the parcel  (+20)",
          "delivered": "parcel delivered  (+100)", "collision": "drove into the forklift lane  (-100)"}


def status_text(episode, t):
    """Status line and last-event line for step t of a recorded episode."""
    carrying = episode["states"][t][2]
    status = (f"Step {t}    Return {sum(episode['rewards'][:t]):+d}    "
              f"{'Carrying the parcel' if carrying else 'No parcel yet'}")
    event = ""
    if t:
        event = EVENTS.get(episode["events"][t - 1], "")
        if episode["slips"][t - 1]:
            event = (event + "  " if event else "") + "(wheels slipped sideways!)"
    return status, event


def result_text(episode, shortest):
    return (f"Episode over: {plots.describe(episode, shortest)}    "
            f"(return {episode['reward']:+d} after {episode['steps']} steps)")


def episode_frames(episode, hold=14):
    """Steps of a recorded episode to draw. Endless loops are cut short, long walks sped up."""
    n = len(episode["states"])
    loop = plots.loop_start(episode["states"])
    last = n - 1 if loop is None else min(n - 1, loop + 8)
    stride = 1 if last <= 60 else 2 if last <= 120 else 3
    steps = list(range(0, last + 1, stride))
    if steps[-1] != last:
        steps.append(last)
    return [(t, False) for t in steps] + [(last, True)] * hold


class RobotView:
    """One warehouse map with an animated robot, its trail and two lines of status text."""

    def __init__(self, ax, env, color):
        self.labels = plots.draw_map(ax, env)
        self.trail, = ax.plot([], [], color=color, lw=2.5, alpha=0.5, zorder=4)
        self.robot = ax.add_patch(Circle(env.start[::-1], 0.3, facecolor=color,
                                         edgecolor=plots.SURFACE, lw=2, zorder=6))
        self.parcel = ax.add_patch(Rectangle((0, 0), 0.26, 0.26, facecolor=plots.PARCEL,
                                             edgecolor=plots.SURFACE, lw=1.5, zorder=7))
        self.title = ax.set_title("", fontsize=12)
        self.status = ax.text(-0.45, env.n_rows - 0.32, "", fontsize=10.5, color=plots.INK, va="top")
        self.event = ax.text(-0.45, env.n_rows + 0.22, "", fontsize=10, color=plots.INK_2, va="top")

    def show(self, episode, t, note=None):
        """Draw the robot as it was after t steps of a recorded episode."""
        states = episode["states"][: t + 1]
        r, c, carrying = states[-1]
        self.trail.set_data([s[1] for s in states], [s[0] for s in states])
        self.robot.center = (c, r)
        self.parcel.set_visible(bool(carrying))
        self.parcel.set_xy((c + 0.02, r - 0.42))
        self.labels["P"].set_alpha(0.3 if carrying else 1.0)
        status, event = status_text(episode, t)
        self.status.set_text(status)
        self.event.set_text(note if note is not None else event)


def has_window():
    return matplotlib.get_backend().lower() not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def play(fig, update, frames, args, default_name, save_count=None):
    """Show the animation in a window, or save it as a GIF (with --save, or when there's no display)."""
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, repeat=False,
                         cache_frame_data=False, save_count=save_count)
    if args.save or not has_window():
        path = Path(args.save) if args.save else results_dir(args.slip) / default_name
        if not args.save:
            print("No window available: there is no display, or matplotlib has no GUI toolkit.\n"
                  "(On Ubuntu, fix this once with: sudo apt install python3-tk)\n"
                  "Saving the animation as a GIF and opening it in your browser instead.")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {path} (this takes a little while)...")
        anim.save(path, writer=PillowWriter(fps=args.fps), dpi=90)
        print("Saved.")
        if not args.save:
            webbrowser.open(path.resolve().as_uri())
    else:
        plt.show()


# ---------------------------------------------------------------- the three demos

def story_stages(models, args):
    """Three snapshots of the Q-learning robot: untrained, part-way, fully trained."""
    snapshots = models["Q-learning"]
    final = max(snapshots)
    rollouts = {ep: greedy_rollout(q, args.slip) for ep, q in snapshots.items()}
    if args.stages:
        chosen = [final if s == "final" else int(s) for s in args.stages.split(",")]
    else:
        # Part-way: the first snapshot where it can fetch the parcel but not yet deliver it.
        middle = [ep for ep in rollouts if 0 < ep < final]
        partial = [ep for ep in middle if rollouts[ep]["outcome"] != "delivered"
                   and any(s[2] for s in rollouts[ep]["states"])]
        chosen = [0, partial[0] if partial else middle[len(middle) // 2], final]
    stages = []
    for ep in chosen:
        if ep not in rollouts:
            raise SystemExit(f"No snapshot after {ep} episodes. Available: {sorted(rollouts)}")
        title = ("Before training: the robot knows nothing yet" if ep == 0 else
                 f"After {ep} training episodes" + (": fully trained" if ep == final else ""))
        stages.append((title, rollouts[ep]))
    return stages


def story(env, models, shortest, args):
    stages = story_stages(models, args)
    if args.ascii:
        return ascii_story(env, stages, shortest, args.fps)
    fig, ax = plt.subplots(figsize=(7, 7.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.1)
    fig.suptitle("Warehouse robot learning with Q-learning", x=0.03, ha="left",
                 fontsize=14, fontweight="bold")
    view = RobotView(ax, env, plots.AGENT_COLORS["Q-learning"])
    frames = [(i, t, final) for i, (_, ep) in enumerate(stages) for t, final in episode_frames(ep)]

    def update(frame):
        i, t, final = frame
        title, episode = stages[i]
        view.title.set_text(f"{i + 1}/{len(stages)}   {title}")
        view.show(episode, t, note=result_text(episode, shortest) if final else None)

    play(fig, update, frames, args, "demo_story.gif")


def compare(env, models, shortest, args):
    final = max(models["Q-learning"])
    rollouts = {name: greedy_rollout(models[name][final], args.slip) for name in plots.LEARNERS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.11, wspace=0.06)
    fig.suptitle("Same warehouse, same rewards: Q-learning vs. SARSA", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.02, 0.905, "Q-learning (off-policy) learns the best route assuming it never makes a "
             "random move.  SARSA (on-policy) learns a route that stays safe even with its own "
             "random exploration.", fontsize=10, color=plots.INK_2)
    views = {}
    for ax, name in zip(axes, plots.LEARNERS):
        views[name] = RobotView(ax, env, plots.AGENT_COLORS[name])
        views[name].title.set_text(name)
    length = max(len(r["states"]) for r in rollouts.values())
    frames = list(range(length)) + [length - 1] * 20

    def update(t):
        for name, view in views.items():
            episode = rollouts[name]
            last = len(episode["states"]) - 1
            view.show(episode, min(t, last), note=result_text(episode, shortest) if t >= last else None)

    play(fig, update, frames, args, "demo_compare.gif")


def live(env, shortest, args):
    """Train a brand-new Q-learning agent on screen."""
    agent = make_agent(QLearningAgent, env, seed=10_000)
    test_env = make_env(args.slip, seed=50_000)
    test_rng = np.random.default_rng(90_000)
    episodes, chunk = episodes_for(args.slip), 10
    color = plots.AGENT_COLORS["Q-learning"]
    best = best_possible_return(shortest)
    random_return = random_baseline(args.slip)
    rewards = []

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.02, right=0.97, top=0.86,
                          bottom=0.12, wspace=0.2)
    ax_map, ax_curve = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle("Watch it learn: a brand-new Q-learning robot, trained live", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    view = RobotView(ax_map, env, color)
    raw, = ax_curve.plot([], [], color=color, lw=0.8, alpha=0.25)
    avg, = ax_curve.plot([], [], color=color, lw=2.2, label=f"{plots.WINDOW}-episode average")
    ax_curve.set_xlim(0, episodes)
    ax_curve.set_ylim(min(-700, random_return - 60), best + 40)
    plots._reference(ax_curve, best, f"best possible: {best}")
    plots._reference(ax_curve, random_return, f"random agent: {random_return:.0f}", color=plots.MUTED)
    ax_curve.set_title("Total reward per training episode")
    ax_curve.set_xlabel("Training episode")
    ax_curve.set_ylabel("Return (total reward)")
    ax_curve.legend(loc="lower right")

    def current_route():
        return test_policy(agent, test_env, 1, test_rng, record=True)[0]

    def frames():
        for _ in range(0, episodes, chunk):
            for _ in range(chunk):
                rewards.append(run_episode(env, agent)["reward"])
                agent.end_episode()
            yield "train", current_route(), None
        final_route = current_route()
        for t, final in episode_frames(final_route, hold=25):
            yield "final", final_route, (t, final)

    def update(frame):
        kind, route, step = frame
        if kind == "train":
            x = np.arange(1, len(rewards) + 1)
            raw.set_data(x, rewards)
            if len(rewards) >= plots.WINDOW:
                avg.set_data(x[plots.WINDOW - 1:], plots._moving_average(np.array(rewards)))
            view.title.set_text(f"Training episode {len(rewards)} of {episodes}    "
                                f"exploration ε = {agent.epsilon:.2f}")
            view.show(route, len(route["states"]) - 1,
                      note=f"Its current best route: {plots.describe(route, shortest)}")
        else:
            t, final = step
            view.title.set_text("Training finished: following the learned route")
            view.show(route, t, note=result_text(route, shortest) if final else None)

    play(fig, update, frames, args, "demo_live.gif", save_count=episodes // chunk + 400)


def ascii_story(env, stages, shortest, fps):
    for i, (title, episode) in enumerate(stages):
        for t, final in episode_frames(episode, hold=1):
            status, event = status_text(episode, t)
            print("\033[2J\033[H", end="")
            print(f"{i + 1}/{len(stages)}  {title}\n")
            print(env.render_ascii(episode["states"][t]))
            print(f"\n{status}\n{result_text(episode, shortest) if final else event}")
            print("\nr = robot, R = robot carrying the parcel, # = rack, ~ = forklift lane")
            time.sleep(1 / fps)
        time.sleep(2.5)


def random_baseline(slip):
    """Average return of the random agent, from the last training run's summary if there is one."""
    summary = results_dir(slip) / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text())["agents"]["Random"]["test_avg_return"]["mean"]
    env, agent = make_env(slip, seed=123), RandomAgent(seed=456)
    return float(np.mean([run_episode(env, agent, learn=False)["reward"] for _ in range(200)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true", help="Q-learning and SARSA side by side")
    mode.add_argument("--live", action="store_true", help="train a new agent live on screen")
    mode.add_argument("--ascii", action="store_true", help="play the story in the terminal")
    parser.add_argument("--save", metavar="FILE.gif", help="save the animation instead of showing it")
    parser.add_argument("--slip", type=float, default=0.0, help="use the slippery-floor models")
    parser.add_argument("--fps", type=float, default=8, help="animation speed (frames per second)")
    parser.add_argument("--stages", help="story snapshots to show, e.g. 0,150,final")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")  # no window needed when writing a file
    plots.apply_style()
    env = make_env(args.slip, seed=0)
    shortest = env.shortest_route_length()
    if args.live:
        live(env, shortest, args)
        return
    models = load_models(args.slip)
    if args.compare:
        compare(env, models, shortest, args)
    else:
        story(env, models, shortest, args)


if __name__ == "__main__":
    main()
