# AlphaGo-Zero-Compact
A compact Python implementation of AlphaGo Zero (9x9 Go), based on the `dlgo`
code from *Deep Learning and the Game of Go*.

## Setup
Requires **Python 3.10** (TensorFlow 2.10 is the last release with native
Windows GPU support). With [uv](https://docs.astral.sh/uv/):

```
uv venv --python 3.10 .venv
uv pip install --python .venv -r requirements.txt
```

### GPU (optional, NVIDIA)
Install the CUDA 11.8 / cuDNN 8.9 runtime as pip wheels in the same venv; no
system CUDA install is needed, only an NVIDIA driver:

```
uv pip install --python .venv -r requirements-gpu.txt
```

`dlgo` puts these libraries on the DLL search path automatically (import `dlgo`
before `tensorflow`). Check with
`python -c "import dlgo, tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
To force the CPU, set `CUDA_VISIBLE_DEVICES=-1`. Worker processes share the GPU
via memory growth.

All commands below are run from the `DLGo/` directory with that environment's
Python (`../.venv/Scripts/python` on Windows, `../.venv/bin/python` elsewhere).

## Create a new agent
```
python init_zero_agent_v2.py bots/my_agent.hdf5          # 9x9 by default
```

## Train
```
python play_train_eval_zero.py --lagent bots/my_agent.hdf5 --ragent bots/my_agent.hdf5 \
    -d work -l work/log -g 1000 -w 7 --gpu-workers 1 -p 32
```
Self-play is batched: each worker plays `-p/--parallel-games` games at once and
evaluates all their search leaves in one network call. The search itself is
Python-bound, so use one GPU worker (`--gpu-workers 1`; several processes
sharing one GPU are slower on Windows) plus single-threaded CPU workers on the
remaining cores (`-w` = total workers). Workers take batches of games from a
shared queue, so faster workers simply play more games.
With `--inference-server` one process owns the GPU and does all network work
for the `-w` search workers: each search round, every worker sends the encoded
leaves of all its games (both agents) in one request, and the server waits for
all workers, merges them into one batch per model and applies the board
symmetries, the network and the prior normalisation on the GPU. The workers
only run the tree search (`--gpu-workers` is ignored). A GPU call costs about
the same for 8 or 256 positions, so give it big batches: on a Ryzen 7 3700X +
RTX 3080, `-w 4 -p 64` (about 124 positions per call) plays 83 moves/s at 20%
CPU, versus 39 moves/s at 47% CPU for the hybrid `-w 7 --gpu-workers 1 -p 16`.

Each iteration plays `--rounds-per-eval` (default 10) rounds of `-g` self-play
games, trains on each round, then plays `--eval-games` (default 100) against the
reference agent; at >= 55% wins the learner becomes the new reference
(saved as `work/zero_agent_v2_<games>.hdf5`). `--max-iterations N` stops after N
evaluations (default: run forever).

## Agents trained with the old (TF 2.2 / Keras 2.3) version
Convert them once so they run on TF 2.10 (outputs are identical):
```
python convert_legacy_zero_agent.py bots/zeroagent_v2.hdf5 bots/zeroagent_v2_tf2.hdf5
```

## Code map
Open `schematics/rlgo_code_map.html` in a browser for diagrams of the training loop,
the batched search, the library layers and the deployment paths, plus a function
reference.

## Play against the bot
```
python web_demo.py            # serves the highest-rated generation of the latest run
```
Open http://127.0.0.1:5000/ and click the board to play Black. Options:
`--agent <file.hdf5>`, `--rounds 400` (stronger, slower), `--gpu`, `--port`.
In the browser: **Undo** takes back your last move and the bot's reply, **Pass**
passes, and the bar shows each side's estimated win chance from the bot's search.
The bot resigns when its estimated win chance drops below 5% after move 30
(`--resign-threshold`, `--resign-min-moves`; `--resign-threshold 0` never resigns).

## Training progress
Run with `--report-dir ../schematics` and the training loop copies its Elo
chart to `schematics/rlgo_elo_report.html` (and the ratings to
`schematics/rlgo_elo.csv`) after every iteration. Gating matches against the
reference are counted in the ratings along with the Elo matches.

Each new generation plays `--elo-games` games against the previous
`--elo-opponents` generations and one checkpoint opponent: at iteration *k*,
the highest-rated of the first 2^n generations for the largest 2^n < *k*
(iteration 188 plays the best of gen_0000..gen_0127). The window doubles over
time, so the checkpoint is a strong yardstick that rarely changes, unlike the
random gen 0, which every trained generation beats 40-0 and which therefore
cannot tell how much stronger it is. Generation 0 remains the 0-Elo anchor;
`--elo-checkpoint gen0` restores the old behaviour.

For a run rated the old way, backfill those checkpoint matches once (with
training stopped, since both need the GPU):
```
python backfill_elo.py -d work/scratch2 -l work/scratch2/log --every 8 --games 40 --report-dir ../schematics
```

## Run in the background on Windows
Double-click `DLGo\start_training.bat` to resume the training run in its own
window, and `DLGo\start_demo.bat` to play the best bot in the browser. Close a
window to stop it; starting it again continues where it left off. Start them
yourself rather than from Claude Code: processes started from a Claude Code
session are stopped when its turn ends.
