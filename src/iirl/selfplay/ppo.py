"""PPO self-play training (model-free, no search), vectorized over many simultaneous games.

Why PPO here: CFR needs a short horizon and AlphaZero needs a learnable value; a long, high-variance
imperfect-info game like Ventures has neither. Model-free self-play does not care — it samples whole
episodes, and the value network is only a variance-reduction *baseline*, not a search leaf-eval.

What makes it actually learn on a ~150-decision game (the first version, terminal-reward-only with
1-sample Monte-Carlo credit, plateaued well below the heuristic):

* **Potential-based reward shaping.** If the game provides ``heuristic_value`` (Ventures' deck-clock
  potential), each decision is rewarded with the change in that potential until the player's next
  decision, and the final step with ``return - potential``. The shaped rewards telescope to the true
  return minus a policy-independent constant, so the optimal policy is unchanged (Ng et al. 1999) —
  but credit lands on the move that caused it instead of being smeared over ~75 moves.
* **GAE(λ)** advantages over each player's own decision sequence, on top of that dense signal.
* **Scale.** Rollouts step ``num_envs`` games in lockstep so each policy call is one batched
  forward pass, optionally across ``num_workers`` processes; 100k+ training games are cheap.
* **Opponent pool.** A fraction of games pits the learner against frozen past snapshots, which damps
  the strategy cycling pure current-vs-current self-play is prone to.

Progress is Elo vs fixed anchors (greedy policy), plus the mean score margin vs each anchor, which
is a far lower-variance signal than win rate on this game.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.runner import _sample_chance
from iirl.eval.tournament import anchored_elo
from iirl.nn.policy_value import PolicyValueNet


@dataclass
class PPOConfig:
    """Hyperparameters for a PPO self-play run."""

    iterations: int = 40
    games_per_iter: int = 64
    num_envs: int = 64
    epochs: int = 4
    minibatch_size: int = 1024
    lr: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    shaping: bool = True
    reward_scale: float = 50.0
    win_bonus: float = 0.0  # points added to the final return for winning (Elo counts wins)
    hidden: tuple[int, ...] = (512, 512)
    num_workers: int = 0  # rollout processes; 0 collects in-process
    privileged_critic: bool = False  # value baseline also sees the opponent's view (train-only)
    past_opponent_frac: float = 0.2
    # Pool opponents act greedily (their argmax move) rather than sampling. Use it to train a
    # best-response exploiter against a deployed greedy agent: the exploiter must face the policy
    # as it is actually played.
    opponent_greedy: bool = False
    # Magnetic regularization (MMD / R-NaD-style): add ``magnet_coef * KL(pi || magnet)`` to the
    # loss, where the magnet is a frozen copy of the policy refreshed every ``magnet_interval``
    # iterations. Pulling each update toward a slowly moving reference damps the strategy cycling
    # of plain self-play, which in two-player zero-sum games lets the iterates converge toward a
    # Nash equilibrium instead of orbiting it (Sokota et al. 2023; Perolat et al. 2022). 0 is off.
    magnet_coef: float = 0.0
    magnet_interval: int = 20
    # Iterations at the start that train only the value baseline (no policy update). A warm-started
    # policy with a fresh critic would otherwise take its first steps on noisy advantages, which
    # can make a strong policy worse before the critic catches up (e.g. a best-response exploiter).
    critic_warmup: int = 0
    snapshot_interval: int = 10
    max_pool: int = 20
    # League play: this fraction of games is against fixed scripted agents (e.g. a heuristic),
    # which adds strategic diversity self-play alone lacks. Only the learner's seat trains.
    scripted_opponents: tuple[Agent, ...] = ()
    scripted_frac: float = 0.0
    eval_interval: int = 4
    eval_games: int = 80
    feature_sample_games: int = 40
    seed: int = 0
    anchor_elos: Mapping[str, float] = field(
        default_factory=lambda: {"random": 1000.0, "heuristic": 1676.0}
    )


@dataclass
class PPOEvalPoint:
    """One point on the PPO training curve."""

    iteration: int
    games: int
    elo: float
    win_rate_vs: dict[str, float]
    policy_loss: float
    value_loss: float
    margin_vs: dict[str, float] = field(default_factory=dict)


def _feature_stats(game: Game, rng: np.random.Generator, num_games: int):
    """Estimate feature mean/std from random self-play (to standardize the net's inputs)."""
    rows = []
    for _ in range(num_games):
        state = game.new_initial_state()
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                state = game.apply_action(state, _sample_chance(game, state, rng))
                continue
            rows.append(game.information_state_tensor(state, game.current_player(state)))
            state = game.apply_action(state, int(rng.choice(game.legal_actions(state))))
    arr = np.asarray(rows, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def _advance_chance(game: Game, state: State, rng: np.random.Generator) -> State:
    while game.is_chance_node(state):
        state = game.apply_action(state, _sample_chance(game, state, rng))
    return state


def _policy_batch(
    net: PolicyValueNet, feats: np.ndarray, masks: np.ndarray, greedy: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sampled (or greedy) actions, their log-probs, and values for a batch of states."""
    with torch.no_grad():
        logits, values = net(torch.as_tensor(feats))
        logits = logits.masked_fill(torch.as_tensor(masks) == 0, -1e9)
        logp_all = torch.log_softmax(logits, dim=1)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    return actions.numpy(), logp.numpy(), values.numpy()


class _Env:
    """One in-flight game: its state, who controls each seat, and per-seat trajectories."""

    def __init__(self, state: State, seats: tuple[object, object], learn: tuple[bool, bool]):
        self.state = state
        self.seats = seats
        self.learn = learn
        self.traj: list[dict[str, list]] = [
            {"feat": [], "mask": [], "action": [], "logp": [], "value": [], "phi": []}
            for _ in range(2)
        ]


def _phi(game: Game, state: State, player: int, shaping: bool) -> float:
    if not shaping:
        return 0.0
    value = game.heuristic_value(state, player)
    return 0.0 if value is None else float(value)


class _Critic(nn.Module):
    """A value-only MLP over privileged (training-time only) features."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp((x - self.feat_mean) / self.feat_std).squeeze(-1)


def _privileged(game: Game, state: State, player: int, own: np.ndarray) -> np.ndarray:
    """Critic input: the player's own view plus the opponent's (their hidden hand included).

    Only the value baseline sees this, and only during training — the policy never does — so the
    learned agent still acts on its own information alone (an asymmetric actor-critic).
    """
    return np.concatenate([own, game.information_state_tensor(state, 1 - player)])


def _run_envs(
    game: Game,
    envs: list[_Env],
    rng: np.random.Generator,
    shaping: bool,
    greedy: bool = False,
    critic: _Critic | None = None,
    record_privileged: bool = False,
    opponent_greedy: bool = False,
) -> list[list[float]]:
    """Step every env to terminal in lockstep, batching each seat-controller's forward passes.

    A seat is either a :class:`PolicyValueNet` (batched) or an :class:`Agent` (stepped one by one).
    Learner seats record their trajectory into ``env.traj``; with a ``critic``, their values come
    from it (on privileged features) instead of the policy net's own value head;
    ``record_privileged`` records those features (``traj["cfeat"]``) even without a critic. Returns
    each env's terminal returns. ``opponent_greedy`` makes net seats with no learning seat among
    their games (frozen opponents) act greedily.
    """
    num_actions = game.num_distinct_actions()
    returns: list[list[float] | None] = [None] * len(envs)
    active = list(range(len(envs)))
    while active:
        groups: dict[int, list[tuple[int, int]]] = {}
        still = []
        for i in active:
            env = envs[i]
            env.state = _advance_chance(game, env.state, rng)
            if game.is_terminal(env.state):
                returns[i] = game.returns(env.state)
                continue
            still.append(i)
            player = game.current_player(env.state)
            ctrl = env.seats[player]
            if isinstance(ctrl, Agent):
                env.state = game.apply_action(env.state, ctrl.step(game, env.state, rng))
            else:
                groups.setdefault(id(ctrl), []).append((i, player))
        active = still
        for members in groups.values():
            net = envs[members[0][0]].seats[members[0][1]]
            feats = np.zeros((len(members), net.input_dim), dtype=np.float32)
            masks = np.zeros((len(members), num_actions), dtype=np.float32)
            for row, (i, player) in enumerate(members):
                feats[row] = game.information_state_tensor(envs[i].state, player)
                masks[row, game.legal_actions(envs[i].state)] = 1.0
            frozen = not any(envs[i].learn[pl] for i, pl in members)
            actions, logps, values = _policy_batch(
                net, feats, masks, greedy or (opponent_greedy and frozen)
            )
            cfeats: dict[int, np.ndarray] = {}
            if critic is not None or record_privileged:
                rows = [r for r, (i, pl) in enumerate(members) if envs[i].learn[pl]]
                if rows:
                    stacked = np.stack(
                        [
                            _privileged(game, envs[members[r][0]].state, members[r][1], feats[r])
                            for r in rows
                        ]
                    )
                    for k, r in enumerate(rows):
                        cfeats[r] = stacked[k]
                    if critic is not None:
                        with torch.no_grad():
                            cvals = critic(torch.as_tensor(stacked)).numpy()
                        for k, r in enumerate(rows):
                            values[r] = cvals[k]
            for row, (i, player) in enumerate(members):
                env = envs[i]
                if env.learn[player]:
                    t = env.traj[player]
                    if row in cfeats:
                        t.setdefault("cfeat", []).append(cfeats[row])
                    t["feat"].append(feats[row])
                    t["mask"].append(masks[row])
                    t["action"].append(int(actions[row]))
                    t["logp"].append(float(logps[row]))
                    t["value"].append(float(values[row]))
                    t["phi"].append(_phi(game, env.state, player, shaping))
                env.state = game.apply_action(env.state, int(actions[row]))
    return returns  # type: ignore[return-value]


def _gae(
    phi: list[float], values: list[float], final_return: float, config: PPOConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (advantages, value targets) for one player's decision sequence (γ = 1).

    Rewards are potential differences between the player's consecutive decisions, and the final
    step collects ``return - phi``; the sum telescopes to ``return - phi[0]``.
    """
    n = len(values)
    phi_next = np.append(np.asarray(phi[1:], dtype=np.float64), final_return)
    rewards = (phi_next - np.asarray(phi, dtype=np.float64)) / config.reward_scale
    v = np.asarray(values, dtype=np.float64)
    v_next = np.append(v[1:], 0.0)
    deltas = rewards + v_next - v
    adv = np.zeros(n)
    running = 0.0
    for k in range(n - 1, -1, -1):
        running = deltas[k] + config.gae_lambda * running
        adv[k] = running
    return adv, adv + v


_BATCH_KEYS = ("feat", "mask", "action", "logp", "adv", "target")


def _collect(
    game: Game,
    net: PolicyValueNet,
    pool: list[PolicyValueNet],
    num_games: int,
    rng: np.random.Generator,
    config: PPOConfig,
    critic: _Critic | None = None,
) -> dict[str, np.ndarray]:
    """Play ``num_games`` training games; return a batch of the learner's decisions.

    Each game is self-play, or (per the config's fractions) the learner vs a scripted opponent or
    vs a pooled past snapshot, with only the learner's seat recorded.
    """
    net.eval()
    keys = _BATCH_KEYS + (("cfeat",) if critic is not None else ())
    batch: dict[str, list] = {k: [] for k in keys}
    remaining = num_games
    while remaining > 0:
        envs = []
        for _ in range(min(remaining, config.num_envs)):
            if config.scripted_opponents and rng.random() < config.scripted_frac:
                opp = config.scripted_opponents[int(rng.integers(len(config.scripted_opponents)))]
                seat = int(rng.integers(2))
                seats = (net, opp) if seat == 0 else (opp, net)
                learn = (seat == 0, seat == 1)
            elif pool and rng.random() < config.past_opponent_frac:
                past = pool[int(rng.integers(len(pool)))]
                seat = int(rng.integers(2))
                seats = (net, past) if seat == 0 else (past, net)
                learn = (seat == 0, seat == 1)
            else:
                seats, learn = (net, net), (True, True)
            envs.append(_Env(game.new_initial_state(), seats, learn))
        remaining -= len(envs)
        rets_all = _run_envs(
            game, envs, rng, config.shaping, critic=critic, opponent_greedy=config.opponent_greedy
        )
        for env, rets in zip(envs, rets_all, strict=True):
            for player in (0, 1):
                t = env.traj[player]
                if not env.learn[player] or not t["value"]:
                    continue
                final = rets[player] + config.win_bonus * float(np.sign(rets[player]))
                adv, target = _gae(t["phi"], t["value"], final, config)
                for key in keys:
                    if key not in ("adv", "target"):
                        batch[key] += t[key]
                batch["adv"] += list(adv)
                batch["target"] += list(target)
    out = {
        "feat": np.stack(batch["feat"]).astype(np.float32),
        "mask": np.stack(batch["mask"]).astype(np.float32),
        "action": np.asarray(batch["action"], dtype=np.int64),
        "logp": np.asarray(batch["logp"], dtype=np.float32),
        "adv": np.asarray(batch["adv"], dtype=np.float32),
        "target": np.asarray(batch["target"], dtype=np.float32),
    }
    if critic is not None:
        out["cfeat"] = np.stack(batch["cfeat"]).astype(np.float32)
    return out


def _new_net(game: Game, input_dim: int, config: PPOConfig) -> PolicyValueNet:
    return PolicyValueNet(
        input_dim, game.num_distinct_actions(), config.hidden, value_activation="linear"
    )


def _new_critic(input_dim: int, config: PPOConfig) -> _Critic | None:
    return _Critic(2 * input_dim, config.hidden) if config.privileged_critic else None


def _worker_main(conn, game: Game, input_dim: int, config: PPOConfig) -> None:
    """Rollout worker: hold a copy of the learner and pool, collect batches on request."""
    torch.set_num_threads(1)
    net = _new_net(game, input_dim, config)
    critic = _new_critic(input_dim, config)
    pool: dict[int, PolicyValueNet] = {}
    while True:
        msg = conn.recv()
        if msg is None:
            return
        state_dict, critic_dict, new_pool, pool_ids, num_games, seed = msg
        net.load_state_dict(state_dict)
        if critic is not None:
            critic.load_state_dict(critic_dict)
            critic.eval()
        for pid, sd in new_pool.items():
            pool[pid] = _new_net(game, input_dim, config)
            pool[pid].load_state_dict(sd)
            pool[pid].eval()
        for pid in [p for p in pool if p not in pool_ids]:
            del pool[pid]
        rng = np.random.default_rng(seed)
        pooled = [pool[p] for p in pool_ids]
        conn.send(_collect(game, net, pooled, num_games, rng, config, critic))


class _WorkerPool:
    """Persistent rollout processes; each receives only the pool snapshots it has not seen."""

    def __init__(self, game: Game, input_dim: int, config: PPOConfig) -> None:
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        self.sent: list[set[int]] = []
        for _ in range(config.num_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_main, args=(child, game, input_dim, config), daemon=True
            )
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
            self.sent.append(set())

    def collect(
        self,
        net: PolicyValueNet,
        critic: _Critic | None,
        pool: dict[int, PolicyValueNet],
        num_games: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        state_dict = {k: v.clone() for k, v in net.state_dict().items()}
        critic_dict = None if critic is None else critic.state_dict()
        n = len(self.conns)
        shares = [num_games // n + (1 if i < num_games % n else 0) for i in range(n)]
        for i, conn in enumerate(self.conns):
            new = {pid: p.state_dict() for pid, p in pool.items() if pid not in self.sent[i]}
            self.sent[i] = set(pool)
            seed = int(rng.integers(2**31))
            conn.send((state_dict, critic_dict, new, list(pool), shares[i], seed))
        parts = [self._recv(i) for i in range(n)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def _recv(self, i: int) -> dict[str, np.ndarray]:
        """Wait for worker ``i``'s batch, failing loudly (not hanging) if the worker has died."""
        while not self.conns[i].poll(1.0):
            if not self.procs[i].is_alive():
                raise RuntimeError(f"rollout worker {i} exited (code {self.procs[i].exitcode})")
        return self.conns[i].recv()

    def close(self) -> None:
        for conn, proc in zip(self.conns, self.procs, strict=True):
            if proc.is_alive():
                conn.send(None)
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def _ppo_update(
    net: PolicyValueNet,
    critic: _Critic | None,
    opt: optim.Optimizer,
    batch: dict,
    config: PPOConfig,
    rng,
    magnet: PolicyValueNet | None = None,
    value_only: bool = False,
) -> tuple[float, float]:
    """Run PPO epochs over the collected batch; return mean (policy_loss, value_loss).

    With a ``magnet`` (and ``config.magnet_coef > 0``) the loss also pays ``KL(pi || magnet)``
    over the legal actions. ``value_only`` trains just the value baseline (critic warm-up); with no
    separate privileged critic the value head shares the policy's trunk, so that still moves the
    policy slightly.
    """
    feats = torch.as_tensor(batch["feat"])
    cfeats = torch.as_tensor(batch["cfeat"]) if critic is not None else None
    actions = torch.as_tensor(batch["action"])
    masks = torch.as_tensor(batch["mask"])
    old_logp = torch.as_tensor(batch["logp"])
    advantages = torch.as_tensor(batch["adv"])
    targets = torch.as_tensor(batch["target"])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    params = list(net.parameters()) + ([] if critic is None else list(critic.parameters()))

    n = len(actions)
    p_losses, v_losses = [], []
    for _ in range(config.epochs):
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
        for start in range(0, n, config.minibatch_size):
            idx = perm[start : start + config.minibatch_size]
            logits, value = net(feats[idx])
            if critic is not None:
                value = critic(cfeats[idx])
            logits = logits.masked_fill(masks[idx] == 0, -1e9)
            logp_all = torch.log_softmax(logits, dim=1)
            new_logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_logp - old_logp[idx])
            adv = advantages[idx]
            surrogate = torch.min(
                ratio * adv, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv
            )
            policy_loss = -surrogate.mean()
            value_loss = nn.functional.mse_loss(value, targets[idx])
            probs = torch.exp(logp_all)
            entropy = -(probs * logp_all * masks[idx]).sum(dim=1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            if value_only:
                loss = config.value_coef * value_loss
            if magnet is not None and config.magnet_coef > 0 and not value_only:
                with torch.no_grad():
                    mag_logits = magnet(feats[idx])[0].masked_fill(masks[idx] == 0, -1e9)
                    mag_logp = torch.log_softmax(mag_logits, dim=1)
                kl = (probs * (logp_all - mag_logp) * masks[idx]).sum(dim=1).mean()
                loss = loss + config.magnet_coef * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            opt.step()
            p_losses.append(float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def evaluate_vs(
    game: Game,
    net: PolicyValueNet,
    opponent: Agent,
    num_games: int,
    rng: np.random.Generator,
    num_envs: int = 64,
) -> tuple[float, float, float, float]:
    """Play the greedy net vs ``opponent``, seat-balanced; return (score, win, margin, stderr).

    ``score`` counts a draw as half a win (the Elo convention); ``margin`` is the mean payoff to the
    net with its standard error.
    """
    net.eval()
    net_returns: list[float] = []
    for start in range(0, num_games, num_envs):
        envs, net_seat = [], []
        for i in range(start, min(num_games, start + num_envs)):
            seat = i % 2
            seats = (net, opponent) if seat == 0 else (opponent, net)
            envs.append(_Env(game.new_initial_state(), seats, (False, False)))
            net_seat.append(seat)
        for rets, seat in zip(
            _run_envs(game, envs, rng, False, greedy=True), net_seat, strict=True
        ):
            net_returns.append(rets[seat])
    arr = np.asarray(net_returns)
    wins, draws = float((arr > 0).mean()), float((arr == 0).mean())
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return wins + 0.5 * draws, wins, float(arr.mean()), stderr


def train_ppo(
    game: Game,
    anchors: Mapping[str, Agent],
    config: PPOConfig,
    on_eval: Callable[[PPOEvalPoint], None] | None = None,
    on_checkpoint: Callable[[PolicyValueNet, PPOEvalPoint], None] | None = None,
    init_net: PolicyValueNet | None = None,
    opponents: list[PolicyValueNet] | None = None,
) -> tuple[PolicyValueNet, list[PPOEvalPoint]]:
    """Train a policy/value net by PPO self-play; return it with its Elo history.

    ``on_eval`` receives each evaluation point; ``on_checkpoint`` additionally receives the net at
    that point (e.g. to save it), so a long run's progress survives an interruption.

    ``init_net`` warm-starts the learner from a copy of that net (weights and feature scaling).
    ``opponents`` fixes the opponent pool to those frozen nets: no learner snapshots are added, and
    ``past_opponent_frac`` of games are played against them. With ``past_opponent_frac = 1`` this
    trains a best response (an *exploiter*) to the given opponents.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    probe = _advance_chance(game, game.new_initial_state(), rng)
    input_dim = len(game.information_state_tensor(probe, game.current_player(probe)))
    net = _new_net(game, input_dim, config)
    if init_net is not None:
        net.load_state_dict(init_net.state_dict())
        mean, std = net.feat_mean.numpy(), net.feat_std.numpy()
    else:
        mean, std = _feature_stats(game, rng, config.feature_sample_games)
        net.set_feature_stats(mean, std)
    critic = _new_critic(input_dim, config)
    params = list(net.parameters())
    if critic is not None:
        critic.feat_mean.copy_(torch.as_tensor(np.concatenate([mean, mean]), dtype=torch.float32))
        critic.feat_std.copy_(
            torch.clamp(torch.as_tensor(np.concatenate([std, std]), dtype=torch.float32), min=1e-6)
        )
        critic.eval()
        params += list(critic.parameters())
    opt = optim.Adam(params, lr=config.lr)
    workers = _WorkerPool(game, input_dim, config) if config.num_workers > 0 else None

    # Fixed opponents take negative ids so they never collide with learner snapshots.
    pool: dict[int, PolicyValueNet] = {-1 - i: op for i, op in enumerate(opponents or [])}
    games_seen = 0
    history: list[PPOEvalPoint] = []
    last_losses = (float("nan"), float("nan"))
    magnet: PolicyValueNet | None = None

    def record_eval(iteration: int) -> None:
        elos, win_rates, margins = [], {}, {}
        for name, anchor in anchors.items():
            score, win, margin, _ = evaluate_vs(game, net, anchor, config.eval_games, rng)
            win_rates[name], margins[name] = win, margin
            elos.append(anchored_elo(score, config.anchor_elos[name]))
        point = PPOEvalPoint(
            iteration, games_seen, float(np.mean(elos)), win_rates, *last_losses, margins
        )
        history.append(point)
        if on_eval is not None:
            on_eval(point)
        if on_checkpoint is not None:
            on_checkpoint(net, point)

    try:
        record_eval(0)
        for iteration in range(1, config.iterations + 1):
            if workers is not None:
                batch = workers.collect(net, critic, pool, config.games_per_iter, rng)
            else:
                pooled = list(pool.values())
                batch = _collect(game, net, pooled, config.games_per_iter, rng, config, critic)
            games_seen += config.games_per_iter
            net.train()
            if config.magnet_coef > 0 and (iteration - 1) % config.magnet_interval == 0:
                magnet = copy.deepcopy(net).eval()  # refresh the reference policy
            warm = iteration <= config.critic_warmup
            last_losses = _ppo_update(net, critic, opt, batch, config, rng, magnet, warm)
            net.eval()
            if critic is not None:
                critic.eval()
            snapshot = config.past_opponent_frac > 0 and opponents is None
            if snapshot and iteration % config.snapshot_interval == 0:
                pool[iteration] = copy.deepcopy(net)
                for old in sorted(pool)[: -config.max_pool]:
                    del pool[old]
            if iteration % config.eval_interval == 0 or iteration == config.iterations:
                record_eval(iteration)
    finally:
        if workers is not None:
            workers.close()

    return net, history
