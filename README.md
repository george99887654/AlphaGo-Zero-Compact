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

All commands below are run from the `DLGo/` directory with that environment's
Python (`../.venv/Scripts/python` on Windows, `../.venv/bin/python` elsewhere).

## Create a new agent
```
python init_zero_agent_v2.py bots/my_agent.hdf5          # 9x9 by default
```

## Train
```
python play_train_eval_zero.py --lagent bots/my_agent.hdf5 --ragent bots/my_agent.hdf5 \
    -d work -l work/log -g 100 -w 2
```
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
