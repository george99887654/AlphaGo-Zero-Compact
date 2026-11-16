"""Play Go against a trained Zero agent in the browser.

    python web_demo.py                       # best-rated gen_*.hdf5 under work/
    python web_demo.py --agent work/scratch2/gen_0003.hdf5 --rounds 400

Then open http://127.0.0.1:5000/ . You play Black; the bot answers as White.
"""
import argparse
import csv
import glob
import os

import h5py
from flask import redirect

from dlgo import httpfrontend
from dlgo import zero


def best_generation(work_glob='work/*'):
    """Highest-rated generation of the most recently updated training run
    (from its elo.csv), or its newest snapshot if it has no ratings yet."""
    runs = [d for d in glob.glob(work_glob) if glob.glob(os.path.join(d, 'gen_*.hdf5'))]
    if not runs:
        return None
    run = max(runs, key=lambda d: max(os.path.getmtime(f) for f in
                                      glob.glob(os.path.join(d, 'gen_*.hdf5'))))
    ratings_file = os.path.join(run, 'elo.csv')
    if os.path.exists(ratings_file):
        with open(ratings_file, newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if os.path.exists(os.path.join(run, r['generation'] + '.hdf5'))]
        if rows:
            best = max(rows, key=lambda r: float(r['elo']))
            return os.path.join(run, best['generation'] + '.hdf5')
    return max(glob.glob(os.path.join(run, 'gen_*.hdf5')), key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='agent .hdf5 file (default: highest-'
                        'rated gen_*.hdf5 of the latest training run)')
    parser.add_argument('--rounds', type=int, default=200,
                        help='MCTS rounds per move (more = stronger, slower)')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='move sampling temperature (low = play the most '
                             'visited move)')
    parser.add_argument('--gpu', action='store_true',
                        help='use the GPU (default CPU, so a training run on '
                             'the GPU is not slowed down)')
    parser.add_argument('--bind-address', default='127.0.0.1')
    parser.add_argument('--port', '-p', type=int, default=5000)
    args = parser.parse_args()

    import tensorflow as tf
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    agent_file = args.agent or best_generation()
    if agent_file is None:
        raise SystemExit('No agent found: pass --agent <file.hdf5> or train '
                         'one with play_train_eval_zero.py --elo-games ...')
    with h5py.File(agent_file, 'r') as f:
        bot = zero.load_zero_agent(f)
    bot.num_rounds = args.rounds
    bot.set_temperature(args.temperature)
    print('Serving %s (%d rounds/move, temperature %.2f) on http://%s:%d/'
          % (agent_file, args.rounds, args.temperature,
             args.bind_address, args.port))

    web_app = httpfrontend.get_web_app({'zero': bot})

    @web_app.route('/')
    def index():
        # Redirect rather than serve, so the page's relative script paths
        # (./jgoboard-latest.js, large/board.js) resolve under /static/.
        return redirect('/static/play_zero_99.html')

    web_app.run(host=args.bind_address, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
