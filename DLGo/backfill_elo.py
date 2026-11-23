"""Backfill checkpoint Elo matches for an existing training run.

Runs that were rated only against generation 0 and their neighbours have
no matches between distant generations, so their ratings drift flat. This
plays, for every --every-th generation k (and the latest one), the
checkpoint match that play_train_eval_zero.py now plays during training:
gen_k against the highest-rated of the first 2^n generations (largest
2^n < k). Generations are processed in order and the ratings are refitted
after every match, exactly as if the matches had been played live. Results
are appended to elo_matches.csv and the report is regenerated.

Stop training first: both need the GPU.

    python backfill_elo.py -d work/scratch2 -l work/scratch2/log \\
        --every 8 --games 40 --report-dir ../schematics
"""
import argparse
import os
import time

import play_train_eval_zero as p
from dlgo import elo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', '-d', required=True)
    parser.add_argument('--log-file', '-l', required=True)
    parser.add_argument('--every', type=int, default=8,
                        help='backfill every N-th generation (plus the latest)')
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--num-workers', '-w', type=int, default=4)
    parser.add_argument('--parallel-games', '-p', type=int, default=10)
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--report-dir')
    args = parser.parse_args()
    if args.games % 2:
        parser.error('--games must be even so both sides play Black and '
                     'White equally often')

    state = p.load_run_state(args.work_dir, args.log_file)
    generations = state['generations']
    matches_file = os.path.join(args.work_dir, 'elo_matches.csv')
    rows = {r['generation']: r for r in elo.write_ratings(
        os.path.join(args.work_dir, 'elo.csv'),
        elo.fit_ratings(elo.load_matches(matches_file),
                        anchor=generations[0]),
        elo.load_matches(matches_file))}
    targets = [k for k in range(1, len(generations))
               if k % args.every == 0 or k == len(generations) - 1]
    done = {(m['player'], m['opponent'])
            for m in elo.load_matches(matches_file)}
    print('Backfilling %d generations: %s' % (len(targets), targets))
    with open(args.log_file, 'a') as logf:
        for k in targets:
            name = generations[k]
            opponent = p.checkpoint_opponent(args.work_dir, generations[:k], k)
            if opponent == name or (name, opponent) in done:
                print('%s: already played %s, skipping' % (name, opponent))
                continue
            t0 = time.time()
            wins = p.evaluate(
                os.path.join(args.work_dir, name + '.hdf5'),
                os.path.join(args.work_dir, opponent + '.hdf5'),
                num_games=args.games, num_workers=args.num_workers,
                board_size=args.board_size,
                parallel_games=args.parallel_games, inference_server=True)
            losses = args.games - wins
            elo.append_match(matches_file, k, rows[name]['total_games'],
                             name, opponent, wins, losses)
            done.add((name, opponent))
            logf.write('Backfill checkpoint match %s vs %s: %d-%d\n' % (
                name, opponent, wins, losses))
            logf.flush()
            print('BACKFILL %s vs %s (best of first %d): %d-%d in %.0f s' % (
                name, opponent, p.checkpoint_window(k), wins, losses,
                time.time() - t0), flush=True)

    ratings = p.write_elo_report(args.work_dir, generations,
                                 state['reference'], args.report_dir)
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:5]
    print('TOP', ', '.join('%s %.0f' % kv for kv in top))
    print('LATEST %s %.0f' % (generations[-1], ratings[generations[-1]]))


if __name__ == '__main__':
    main()
