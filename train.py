"""
Train, test and compare the warehouse robot's learning agents.

    python train.py              # Q-learning vs SARSA, 10 training runs each, then figures
    python train.py --slip 0.1   # the same on a slippery floor (10% of moves drift sideways)

Writes the figures, summary.json and training_history.npz to results/, and the
learned Q-tables (plus snapshots taken during training, for the demo) to models/.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

import plots
from agents import QLearningAgent, RandomAgent, SarsaAgent, TabularAgent
from warehouse_env import WarehouseEnv, best_possible_return

ROOT = Path(__file__).resolve().parent
ALGORITHMS = (QLearningAgent, SarsaAgent)

# Hyperparameters. The README explains why each value was chosen.
CONFIG = {
    "episodes": 1000,        # training episodes per run
    "episodes_slippery": 2000,  # random slips mean more states to learn about, so train longer
    "runs": 10,              # independent training runs (random seeds) per algorithm
    "alpha": 0.1,            # learning rate
    "gamma": 0.99,           # discount factor
    "epsilon": 1.0,          # start out exploring 100% at random...
    "epsilon_decay": 0.995,  # ...and get greedier every episode (minimum reached near episode 600)
    "epsilon_min": 0.05,
    "max_steps": 200,        # an episode is cut off after this many moves
    "test_every": 10,        # test the greedy policy every N training episodes
    "test_episodes": 100,    # test episodes per trained agent in the final evaluation
}
# Q-table snapshots kept from the first run, to show (and demo) how the robot improves.
CHECKPOINTS = (0, 25, 50, 100, 150, 200, 250, 300, 400, 600, 800)


def slug(name):
    return name.lower().replace("-", "_")


def model_dir(slip):
    return ROOT / "models" / (f"slip_{slip:.2f}" if slip else "")


def results_dir(slip):
    return ROOT / "results" / (f"slip_{slip:.2f}" if slip else "")


def episodes_for(slip):
    return CONFIG["episodes_slippery"] if slip else CONFIG["episodes"]


def make_env(slip, seed):
    return WarehouseEnv(slip_prob=slip, max_steps=CONFIG["max_steps"], seed=seed)


def make_agent(agent_cls, env, seed):
    c = CONFIG
    return agent_cls(env.state_shape, env.n_actions, alpha=c["alpha"], gamma=c["gamma"],
                     epsilon=c["epsilon"], epsilon_decay=c["epsilon_decay"],
                     epsilon_min=c["epsilon_min"], seed=seed)


def run_episode(env, agent, learn=True, greedy=False, record=False):
    """Play one episode: the agent-environment interaction loop.

    learn=True:   update the Q-table after every step (training)
    greedy=True:  always take the best known action, no exploration (testing)
    record=True:  also return every state, event and reward, for figures and the demo
    """
    choose = agent.greedy_action if greedy else agent.act
    state = env.reset()
    action = choose(state)
    total_reward = 0
    states, events, rewards, slips = [state], [], [], []
    while True:
        next_state, reward, terminated, truncated, info = env.step(action)
        next_action = choose(next_state)  # SARSA's update needs the next action
        if learn:
            agent.update(state, action, reward, next_state, next_action, terminated)
        total_reward += reward
        if record:
            states.append(next_state)
            events.append(info["event"])
            rewards.append(reward)
            slips.append(info["slipped"])
        state, action = next_state, next_action
        if terminated or truncated:
            break
    result = {"reward": total_reward, "steps": env.steps,
              "outcome": info["event"] if terminated else "timeout"}
    if record:
        result.update(states=states, events=events, rewards=rewards, slips=slips)
    return result


def test_policy(agent, env, episodes, rng, record=False):
    """Play the greedy policy with learning and exploration switched off.

    The agent borrows `rng` for tie-breaking, so testing never alters the training run.
    """
    training_rng, agent.rng = agent.rng, rng
    results = [run_episode(env, agent, learn=False, greedy=True, record=record)
               for _ in range(episodes)]
    agent.rng = training_rng
    return results


def greedy_rollout(q, slip=0.0, seed=0):
    """Record one test episode of the greedy policy stored in Q-table `q`."""
    env = make_env(slip, seed)
    agent = TabularAgent(env.state_shape, env.n_actions, seed=seed)
    agent.q = q
    return run_episode(env, agent, learn=False, greedy=True, record=True)


def train(agent_cls, seed, slip=0.0, episodes=None, keep_checkpoints=False):
    """One complete training run. Returns (agent, history, tests, checkpoints)."""
    env = make_env(slip, seed)
    test_env = make_env(slip, seed + 50_000)
    test_rng = np.random.default_rng(seed + 90_000)
    agent = make_agent(agent_cls, env, seed + 10_000)
    n_tests = 1 if slip == 0 else 10  # with deterministic moves one test episode says it all

    n = episodes or episodes_for(slip)
    history = {key: np.zeros(n) for key in ("reward", "steps", "delivered", "collision", "epsilon")}
    tests = {key: [] for key in ("episode", "reward", "steps", "delivered")}
    checkpoints = {}

    def run_tests(episode):
        results = test_policy(agent, test_env, n_tests, test_rng)
        tests["episode"].append(episode)
        tests["reward"].append(np.mean([r["reward"] for r in results]))
        tests["steps"].append(np.mean([r["steps"] for r in results]))
        tests["delivered"].append(np.mean([r["outcome"] == "delivered" for r in results]))

    for ep in range(n):
        if keep_checkpoints and ep in CHECKPOINTS:
            checkpoints[ep] = agent.q.copy()
        if ep % CONFIG["test_every"] == 0:
            run_tests(ep)
        history["epsilon"][ep] = agent.epsilon
        result = run_episode(env, agent)
        history["reward"][ep] = result["reward"]
        history["steps"][ep] = result["steps"]
        history["delivered"][ep] = result["outcome"] == "delivered"
        history["collision"][ep] = result["outcome"] == "collision"
        agent.end_episode()
    run_tests(n)
    if keep_checkpoints:
        checkpoints[n] = agent.q.copy()
    return agent, history, {key: np.array(v) for key, v in tests.items()}, checkpoints


def save_models(folder, name, agent, checkpoints):
    folder.mkdir(parents=True, exist_ok=True)
    agent.save(folder / f"{slug(name)}.npy")
    np.savez_compressed(folder / f"{slug(name)}_checkpoints.npz",
                        **{f"ep{ep}": q for ep, q in checkpoints.items()})


def load_models(slip=0.0):
    """Q-table snapshots for the demo, {name: {episode: Q}}. Trains them first if missing."""
    folder = model_dir(slip)
    files = {cls.name: folder / f"{slug(cls.name)}_checkpoints.npz" for cls in ALGORITHMS}
    if not all(path.exists() for path in files.values()):
        print("No trained models found, so training them now (takes a few seconds)...")
        for cls in ALGORITHMS:
            agent, _, _, checkpoints = train(cls, seed=0, slip=slip, keep_checkpoints=True)
            save_models(folder, cls.name, agent, checkpoints)
    models = {}
    for name, path in files.items():
        with np.load(path) as data:
            models[name] = dict(sorted((int(key[2:]), data[key]) for key in data.files))
    return models


def reliable_from(test_episodes, test_delivered):
    """First test point after which the greedy policy always delivers (>= 90% of tests)."""
    ok = test_delivered >= 0.9
    for i in range(len(ok)):
        if ok[i:].all():
            return int(test_episodes[i])
    return None


def mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slip", type=float, default=0.0,
                        help="chance that a move drifts sideways (default 0: deterministic floor)")
    parser.add_argument("--runs", type=int, default=CONFIG["runs"],
                        help="training runs (random seeds) per algorithm")
    parser.add_argument("--episodes", type=int, default=None,
                        help=f"training episodes per run (default {CONFIG['episodes']}, "
                             f"or {CONFIG['episodes_slippery']} with --slip)")
    args = parser.parse_args()
    CONFIG["runs"] = args.runs
    slip = args.slip
    episodes = args.episodes or episodes_for(slip)
    out = results_dir(slip)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(slip, seed=0)
    shortest = env.shortest_route_length()
    print(f"Warehouse robot: Q-learning vs SARSA, {CONFIG['runs']} runs x {episodes} "
          f"episodes, slip probability {slip:.2f}")
    print("\n".join(" ".join(row) for row in env.layout))
    print(f"Shortest possible route: {shortest} steps "
          f"(best possible return {best_possible_return(shortest)})\n")

    start_time = time.time()
    summary = {"config": dict(CONFIG, episodes=episodes, slip=slip), "shortest_route_steps": shortest,
               "best_possible_return": best_possible_return(shortest), "agents": {}}
    history, checkpoints = {}, {}
    for cls in ALGORITHMS:
        print(f"Training {cls.name}...", flush=True)
        runs = [train(cls, seed, slip, episodes, keep_checkpoints=(seed == 0))
                for seed in range(CONFIG["runs"])]
        save_models(model_dir(slip), cls.name, runs[0][0], runs[0][3])
        checkpoints[cls.name] = runs[0][3]

        h = {key: np.stack([run[1][key] for run in runs]) for key in runs[0][1]}
        for key in ("reward", "steps", "delivered"):
            h["test_" + key] = np.stack([run[2][key] for run in runs])
        h["test_episode"] = runs[0][2]["episode"]
        history[cls.name] = h

        finals = [test_policy(agent, make_env(slip, 70_000 + seed), CONFIG["test_episodes"],
                              np.random.default_rng(80_000 + seed))
                  for seed, (agent, *_) in enumerate(runs)]
        last = slice(-100, None)
        summary["agents"][cls.name] = {
            "test_success_pct": mean_std([100 * np.mean([r["outcome"] == "delivered" for r in f])
                                          for f in finals]),
            "test_avg_steps": mean_std([np.mean([r["steps"] for r in f]) for f in finals]),
            "test_avg_return": mean_std([np.mean([r["reward"] for r in f]) for f in finals]),
            "train_last100_success_pct": mean_std(100 * h["delivered"][:, last].mean(axis=1)),
            "train_last100_collision_pct": mean_std(100 * h["collision"][:, last].mean(axis=1)),
            "train_last100_return": mean_std(h["reward"][:, last].mean(axis=1)),
            "train_total_collisions": mean_std(h["collision"].sum(axis=1)),
            "reliable_route_from_episode": [reliable_from(h["test_episode"], d)
                                            for d in h["test_delivered"]],
        }

    # Baseline: an agent that never learns, tested the same way (runs x test episodes).
    random_runs = []
    for seed in range(CONFIG["runs"]):
        random_env, random_agent = make_env(slip, 70_000 + seed), RandomAgent(seed=80_000 + seed)
        random_runs.append([run_episode(random_env, random_agent, learn=False)
                            for _ in range(CONFIG["test_episodes"])])
    summary["agents"]["Random"] = {
        "test_success_pct": mean_std([100 * np.mean([r["outcome"] == "delivered" for r in f])
                                      for f in random_runs]),
        "test_collision_pct": mean_std([100 * np.mean([r["outcome"] == "collision" for r in f])
                                        for f in random_runs]),
        "test_avg_steps": mean_std([np.mean([r["steps"] for r in f]) for f in random_runs]),
        "test_avg_return": mean_std([np.mean([r["reward"] for r in f]) for f in random_runs]),
    }
    summary["training_seconds"] = round(time.time() - start_time, 1)
    print_summary(summary)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out / "training_history.npz",
                        **{f"{slug(name)}__{key}": value
                           for name, h in history.items() for key, value in h.items()})

    routes = {name: greedy_rollout(cps[episodes], slip) for name, cps in checkpoints.items()}
    progress = {ep: greedy_rollout(q, slip) for ep, q in checkpoints["Q-learning"].items()}
    matplotlib.use("Agg")  # figures are only saved to files, no window needed
    plots.make_all(env, history, summary, checkpoints, routes, progress, out)
    print(f"\nFigures and summary.json saved in {out.relative_to(ROOT)}/, "
          f"Q-tables in {model_dir(slip).relative_to(ROOT)}/")


def print_summary(summary):
    agents = summary["agents"]

    def fmt(stat, digits=1):
        return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"

    runs, eps = summary["config"]["runs"], summary["config"]["test_episodes"]
    print(f"\nFinal test with exploration off ({eps} episodes per trained agent, "
          f"mean ± std over {runs} runs)")
    print(f"  {'Agent':<12}{'Success':>16}{'Avg steps':>16}{'Avg return':>18}")
    for name, a in agents.items():
        print(f"  {name:<12}{fmt(a['test_success_pct']) + '%':>16}{fmt(a['test_avg_steps']):>16}"
              f"{fmt(a['test_avg_return']):>18}")
    print(f"  Shortest possible route: {summary['shortest_route_steps']} steps, "
          f"return {summary['best_possible_return']}")

    print(f"\nWhile still exploring (last 100 training episodes, "
          f"epsilon = {summary['config']['epsilon_min']})")
    print(f"  {'Agent':<12}{'Success':>16}{'Collisions':>16}{'Avg return':>18}")
    for name in (cls.name for cls in ALGORITHMS):
        a = agents[name]
        print(f"  {name:<12}{fmt(a['train_last100_success_pct']) + '%':>16}"
              f"{fmt(a['train_last100_collision_pct']) + '%':>16}{fmt(a['train_last100_return']):>18}")

    print("\nEpisodes until the greedy route became reliable (per run):")
    for name in (cls.name for cls in ALGORITHMS):
        print(f"  {name:<12}{agents[name]['reliable_route_from_episode']}")
    print(f"\nTraining and testing took {summary['training_seconds']} s")


if __name__ == "__main__":
    main()
