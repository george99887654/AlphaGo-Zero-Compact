# RLGo (AlphaGo-Zero-Compact): notes for Claude Code

AlphaGo Zero for 9x9 Go: self-play with batched MCTS, a policy/value ResNet,
gating (`--gate-threshold`, 60% of 128 games in the launcher), checkpoint Elo ratings, an opponent league (curated checkpoints, `league_checkpoints.json`) and a browser demo.
User docs are in `README.md`; the architecture map is
`schematics/rlgo_code_map.html`.

## Environment
- Python 3.10 venv at `.venv` (TensorFlow 2.10.1 with tf.keras, CUDA 11.8
  from pip wheels). Use `.venv/Scripts/python.exe` on Windows; the system
  `python` may be a Microsoft Store stub that silently does nothing.
- Run commands from `DLGo/`. Set `TF_CPP_MIN_LOG_LEVEL=2`.
- Import `dlgo` before `tensorflow`: it puts the CUDA DLLs on the path.
  Otherwise GPU init fails with `cudaGetDevice() failed`.

## Workflow rules
- Work on `develop`. Every code change gets its own commit named
  `[RLGo vX.Y.Z] <summary>`, bumping the patch number (minor only when the
  user asks). Push only when the user asks.
- Never start training or the web demo from Claude Code: processes started
  from a session are killed when the turn ends. The user starts
  `DLGo/start_training.bat` / `DLGo/start_demo.bat`. Reading logs and
  reports while training runs is fine.
- GPU-heavy work (benchmarks, Elo backfill) needs training stopped. Ask
  the user to close the training window and check that it has stopped.
- Changes to the training scheme (self-play, targets, gating, league) go
  behind opt-in flags. Don't change existing defaults silently.
- When flags or the launcher change, update `README.md`,
  `DLGo/start_training.bat` and this file in the same commit.

## Code layout (DLGo/)
- `play_train_eval_zero.py`: the training loop
  (`self_play_round` -> `train_on_experience` -> `evaluate` (gating) ->
  `run_elo_matches`), worker processes, `--resume`, the inference server
  wiring and the league.
- `dlgo/zero/agent.py`: `ZeroAgent`, `select_moves_multi` (lock-step MCTS
  for several agents), `evaluate_many`, `ZeroTreeNode`, `train`.
- `dlgo/zero/inference.py`: the GPU inference server (batches all workers;
  symmetry, network and normalisation on the GPU).
- `dlgo/elo.py`: Bradley-Terry ratings and the HTML report.
- `backfill_elo.py`, `web_demo.py` (+ `dlgo/httpfrontend/static/play_zero_99.html`).

## Verifying changes
- Search/agent changes: a fixed-seed fingerprint (CPU via
  `CUDA_VISIBLE_DEVICES=-1`, seeded NumPy, moves + root visit counts over
  several moves) must match the previous commit, checked out in a
  temporary `git worktree`.
- Inference changes: the server's priors/values must match local
  `evaluate_encoded` (max abs diff about 1e-7).
- Throughput: use a steady-state benchmark with all game slots kept full.
  Short runs are dominated by the end-of-run tail and mislead.
- Put scratch scripts and logs in `DLGo/work/tmp/` (git-ignored).

## Run data (git-ignored)
- `DLGo/work/scratch2/`: the live run. `log`, `elo_matches.csv`,
  `elo.csv`, `gen_NNNN.hdf5` snapshots, `zero_agent_v2_<games>.hdf5`
  promoted references, `zero_agent_train.hdf5` (the learner).
- `schematics/rlgo_elo_report.html` and `rlgo_elo.csv` are refreshed after
  every iteration.
