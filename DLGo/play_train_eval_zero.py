import argparse
import datetime
import glob
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from dlgo import elo
from dlgo import kerasutil
from dlgo import scoring
from dlgo import zero
from dlgo.goboard_fast import GameState, Player, Point


def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return zero.load_zero_agent(h5file)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:board.num_cols])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
       return 'B'
    return 'W'

def simulate_game(
        black_agent,
        white_agent,
        board_size,):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    num_moves = 0
    while (not game.is_over()) & (num_moves < 2*board_size*board_size):
        if num_moves < 16:
            # Pick randomly.
            agents[game.next_player].set_temperature(1.0)
        else:
            # Favor the best-looking move.
            agents[game.next_player].set_temperature(0.05)
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
        num_moves += 1
        
    print('number of moves: %d'% num_moves)
    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)
    
    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

def eval_simulate_game(
        black_agent,
        white_agent,
        board_size,):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    num_moves = 0
    while (not game.is_over()) & (num_moves < 2*board_size*board_size):
        agents[game.next_player].set_temperature(0.05)
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
        num_moves += 1
        
    print('number of moves: %d'% num_moves)
    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)
    
    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def self_play_temperature(num_moves):
    # Explore in the opening, then favor the best-looking move.
    return 1.0 if num_moves < 16 else 0.05


def eval_temperature(num_moves):
    return 0.05


def play_games_batched(agent1, agent2, num_games, board_size,
                       temperature_fn, collect=False):
    """Play `num_games` games between agent1 and agent2 at the same time.

    agent1 plays black in even-numbered games and white in odd ones. All games
    advance one move per step; each agent searches all the games where it is
    to move together, so its network evaluates their positions in batches.
    Returns (records, agent1_colors, collectors) where collectors is a list of
    (agent1 collector, agent2 collector) per game, or None if not collecting.
    """
    games = [GameState.new_game(board_size) for _ in range(num_games)]
    agent1_colors = [Player.black if i % 2 == 0 else Player.white
                     for i in range(num_games)]
    collectors = None
    if collect:
        collectors = []
        for _ in range(num_games):
            pair = (zero.ZeroExperienceCollector(),
                    zero.ZeroExperienceCollector())
            for c in pair:
                c.begin_episode()
            collectors.append(pair)
    moves = [[] for _ in range(num_games)]
    max_moves = 2 * board_size * board_size

    num_moves = 0
    active = list(range(num_games))
    while active:
        temperature = temperature_fn(num_moves)
        # Decide who is to move in each game *before* anyone moves, so every
        # game advances exactly one ply per step (num_moves = ply count).
        agent1_to_move = {i: games[i].next_player == agent1_colors[i]
                          for i in active}
        for agent_idx, agent in enumerate((agent1, agent2)):
            idx = [i for i in active
                   if agent1_to_move[i] == (agent_idx == 0)]
            if not idx:
                continue
            agent.set_temperature(temperature)
            chosen = agent.select_moves(
                [games[i] for i in idx],
                [collectors[i][agent_idx] for i in idx] if collect else None)
            for i, move in zip(idx, chosen):
                moves[i].append(move)
                games[i] = games[i].apply_move(move)
        num_moves += 1
        active = [i for i in active
                  if not games[i].is_over() and num_moves < max_moves]

    records = []
    for i in range(num_games):
        result = scoring.compute_game_result(games[i])
        records.append(GameRecord(moves=moves[i], winner=result.winner,
                                  margin=result.winning_margin))
        print('Game %d/%d: %d moves, %s, agent %d wins' % (
            i + 1, num_games, len(moves[i]), result,
            1 if result.winner == agent1_colors[i] else 2))
    return records, agent1_colors, collectors


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def configure_worker(use_gpu):
    """Per-process TensorFlow setup for self-play / evaluation workers.

    GPU workers share the GPU via memory growth. CPU workers hide the GPU
    and use a single TF thread each: the MCTS is Python-bound, so several
    single-threaded CPU workers scale across cores, while multiple processes
    sharing the GPU (on Windows) or large TF thread pools slow each other down.
    Must run before TensorFlow does any work in this process.
    """
    import tensorflow as tf
    if use_gpu:
        kerasutil.set_gpu_memory_target(1.0)
    else:
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)


def claim_games(counter, num_games, batch_size):
    """Take up to batch_size of the remaining games from the shared counter.
    Returns (first_game_index, count); count is 0 when all are claimed."""
    with counter.get_lock():
        start = counter.value
        count = max(0, min(batch_size, num_games - start))
        counter.value = start + count
    return start, count


def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games,
                 experience_filename,
                 counter,
                 parallel_games=32,
                 use_gpu=True):
    """Worker: play self-play games in batches of `parallel_games`, taking
    batches from the shared `counter` until `num_games` games have been
    claimed across all workers, and save the experience it collected."""
    configure_worker(use_gpu)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename)
    agent2 = load_agent(agent2_filename)

    all_collectors = []
    while True:
        start, batch = claim_games(counter, num_games, parallel_games or num_games)
        if batch == 0:
            break
        print('[%s worker %d] Simulating games %d-%d of %d...' % (
            'GPU' if use_gpu else 'CPU', os.getpid(),
            start + 1, start + batch, num_games))
        records, colors1, collectors = play_games_batched(
            agent1, agent2, batch, board_size,
            self_play_temperature, collect=True)
        for record, color1, (collector1, collector2) in zip(
                records, colors1, collectors):
            reward1 = 1 if record.winner == color1 else -1
            collector1.complete_episode(reward=reward1)
            collector2.complete_episode(reward=-reward1)
            all_collectors.extend([collector1, collector2])

    all_collectors = [c for c in all_collectors if c.states]
    if not all_collectors:
        return  # this worker got no games; its file stays empty
    experience = zero.combine_experience(all_collectors)
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, board_size, num_workers,
                        parallel_games=32, gpu_workers=1):
    experience_files = []
    workers = []
    counter = multiprocessing.Value('i', 0)
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                num_games,
                filename,
                counter,
                parallel_games,
                i < gpu_workers,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()
    failed = [w.exitcode for w in workers if w.exitcode != 0]
    if failed:
        raise RuntimeError('%d self-play worker(s) failed' % len(failed))

    # Merge experience buffers (workers that got no games left empty files).
    print('Merging experience buffers...')
    buffers = []
    for filename in experience_files:
        if os.path.getsize(filename) == 0:
            continue
        with h5py.File(filename, 'r') as expf:
            buffers.append(zero.load_experience(expf))
    combined_buffer = zero.combine_experience(buffers)
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size):
    kerasutil.set_gpu_memory_target(1.0)
    learning_agent = load_agent(learning_agent)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = zero.load_experience(expf)
    learning_agent.train(exp_buffer, learning_rate=lr, batch_size=batch_size)

    with h5py.File(output_file, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


def train_on_experience(learning_agent, output_file, experience_file,
                        lr, batch_size):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size
        )
    )
    worker.start()
    worker.join()


def play_games(args):
    agent1_fname, agent2_fname, num_games, board_size, gpu_frac = args

    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname)
    agent2 = load_agent(agent2_fname)

    wins, losses = 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
            
        game_record = simulate_game(black_player, white_player, board_size)
        
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses

def eval_worker(agent1_fname, agent2_fname, num_games, board_size,
                counter, results, parallel_games=32, use_gpu=True):
    """Worker: play evaluation games (agent1 vs agent2) in batches claimed
    from the shared counter; put (wins, losses) for agent1 on `results`."""
    configure_worker(use_gpu)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname)
    agent2 = load_agent(agent2_fname)

    wins, losses = 0, 0
    while True:
        start, batch = claim_games(counter, num_games, parallel_games or num_games)
        if batch == 0:
            break
        print('[%s worker %d] Evaluating games %d-%d of %d...' % (
            'GPU' if use_gpu else 'CPU', os.getpid(),
            start + 1, start + batch, num_games))
        records, colors1, _ = play_games_batched(
            agent1, agent2, batch, board_size, eval_temperature)
        for record, color1 in zip(records, colors1):
            if record.winner == color1:
                wins += 1
            else:
                losses += 1
        print('Agent 1 record (this worker): %d/%d' % (wins, wins + losses))
    results.put((wins, losses))


def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size,
             parallel_games=32, gpu_workers=1):
    counter = multiprocessing.Value('i', 0)
    results = multiprocessing.Queue()
    workers = []
    for i in range(num_workers):
        worker = multiprocessing.Process(
            target=eval_worker,
            args=(learning_agent, reference_agent, num_games, board_size,
                  counter, results, parallel_games, i < gpu_workers))
        worker.start()
        workers.append(worker)
    # Collect results before joining (a process with queued data cannot exit
    # until it is read).
    game_results = [results.get() for _ in workers]
    for worker in workers:
        worker.join()

    total_wins = sum(w for w, _ in game_results)
    total_losses = sum(l for _, l in game_results)
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    return total_wins


def save_generation(work_dir, agent_file, iteration):
    name = 'gen_%04d' % iteration
    shutil.copy(agent_file, os.path.join(work_dir, name + '.hdf5'))
    return name


def run_elo_matches(args, generations, training_agent, iteration,
                    total_games, logf):
    """Snapshot the learner as a new generation, play it against
    generation 0 and the most recent generations, and refresh the ratings."""
    name = save_generation(args.work_dir, training_agent, iteration)
    opponents = [generations[0]] + generations[-args.elo_opponents:]
    opponents = list(dict.fromkeys(opponents))      # dedupe, keep order
    matches_file = os.path.join(args.work_dir, 'elo_matches.csv')
    for opp in opponents:
        wins = evaluate(
            os.path.join(args.work_dir, name + '.hdf5'),
            os.path.join(args.work_dir, opp + '.hdf5'),
            num_games=args.elo_games,
            num_workers=args.num_workers,
            board_size=args.board_size,
            parallel_games=args.parallel_games,
            gpu_workers=args.gpu_workers)
        losses = args.elo_games - wins
        elo.append_match(matches_file, iteration, total_games,
                         name, opp, wins, losses)
        logf.write('Elo match %s vs %s: %d-%d\n' % (name, opp, wins, losses))
    generations.append(name)

    matches = elo.load_matches(matches_file)
    ratings = elo.fit_ratings(matches, anchor=generations[0])
    rows = elo.write_ratings(os.path.join(args.work_dir, 'elo.csv'),
                             ratings, matches)
    elo.write_html_report(os.path.join(args.work_dir, 'elo_report.html'),
                          rows, matches)
    print('Elo of %s: %.0f (gen 0 = 0)' % (name, ratings[name]))
    logf.write('Elo of %s after %d games: %.0f\n' % (
        name, total_games, ratings[name]))


def load_run_state(work_dir, log_file, fallback_reference=None):
    """Reconstruct where a run in work_dir stopped, for --resume.

    The learner is work_dir/zero_agent_train.hdf5 as last written. The
    reference is the last "New reference is ..." in the log (falling back to
    the run's starting reference). Completed iterations are the gen_NNNN
    snapshots, and the game count comes from the Elo match history.
    """
    learner = os.path.join(work_dir, 'zero_agent_train.hdf5')
    if not os.path.exists(learner):
        raise SystemExit('Cannot resume: %s not found' % learner)
    reference = None
    first_reference = None
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if line.startswith('New reference is '):
                    reference = line[len('New reference is '):].strip()
                elif line.startswith('Referencing from ') and first_reference is None:
                    first_reference = line[len('Referencing from '):].rsplit(' at ', 1)[0]
    reference = reference or fallback_reference or first_reference
    if not reference or not os.path.exists(reference):
        raise SystemExit('Cannot resume: reference agent %r not found '
                         '(pass --ragent)' % reference)
    gens = sorted(os.path.basename(g)[:-len('.hdf5')]
                  for g in glob.glob(os.path.join(work_dir, 'gen_*.hdf5')))
    iteration = int(gens[-1][len('gen_'):]) if gens else 0
    total_games = 0
    for m in elo.load_matches(os.path.join(work_dir, 'elo_matches.csv')):
        if m['iteration'] == iteration:
            total_games = m['total_games']
    return dict(reference=reference, generations=gens,
                iteration=iteration, total_games=total_games)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lagent', help='starting learner (not needed with --resume)')
    parser.add_argument('--ragent', help='starting reference (not needed with --resume)')
    parser.add_argument('--resume', action='store_true',
                        help='continue the run in --work-dir: keep its learner, '
                             'latest promoted reference, generations and Elo '
                             'history (an interrupted iteration is restarted)')
    parser.add_argument('--games-per-batch', '-g', type=int, default=1000)
    parser.add_argument('--work-dir', '-d', required=True)
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--log-file', '-l', required=True)
    parser.add_argument('--rounds-per-eval', type=int, default=10,
                        help='self-play + train rounds between evaluations')
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='stop after this many evaluations (0 = run forever)')
    parser.add_argument('--parallel-games', '-p', type=int, default=32,
                        help='games each worker plays simultaneously; their '
                             'network evaluations are batched (0 = all at once)')
    parser.add_argument('--elo-games', type=int, default=0,
                        help='after each iteration, play this many games '
                             'against generation 0 and the previous '
                             '--elo-opponents generations and update '
                             'elo.csv / elo_report.html (0 = off)')
    parser.add_argument('--elo-opponents', type=int, default=2)
    parser.add_argument('--gpu-workers', type=int, default=1,
                        help='how many of the --num-workers use the GPU; the '
                             'rest run single-threaded on the CPU')

    args = parser.parse_args()
    if not args.resume and not (args.lagent and args.ragent):
        parser.error('--lagent and --ragent are required unless --resume')
    os.makedirs(args.work_dir, exist_ok=True)

    logf = open(args.log_file, 'a')
    logf.write('----------------------\n')
    experience_file = os.path.join(args.work_dir, 'zero_exp_temp.hdf5')
    tmp_agent = os.path.join(args.work_dir, 'zero_agent_temp.hdf5')
    # training_agent is the single, continuously-updated learner: it plays
    # the self-play games, every training round starts from its latest
    # weights and writes back to it, and it is the agent that is evaluated.
    training_agent = os.path.join(args.work_dir, 'zero_agent_train.hdf5')
    if args.resume:
        state = load_run_state(args.work_dir, args.log_file, args.ragent)
        reference_agent = state['reference']
        generations = state['generations']
        iteration = state['iteration']
        total_games = state['total_games']
        logf.write('Resumed at %s: iteration %d, %d games so far, '
                   'reference %s\n' % (datetime.datetime.now(), iteration,
                                       total_games, reference_agent))
        print('Resuming after iteration %d (%d games), reference %s' % (
            iteration, total_games, reference_agent))
    else:
        logf.write('Starting from %s at %s\n' % (
            args.lagent, datetime.datetime.now()))
        logf.write('Referencing from %s at %s\n' % (
            args.ragent, datetime.datetime.now()))
        reference_agent = args.ragent
        total_games = 0
        iteration = 0
        shutil.copy(args.lagent, training_agent)
        generations = []      # snapshot names, in order, for Elo matches
        if args.elo_games:
            generations.append(save_generation(args.work_dir, training_agent, 0))
    start_iteration = iteration
    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n' % (total_games,))
        for i in range(args.rounds_per_eval):
            print('Round %d/%d' % (i + 1, args.rounds_per_eval))
            generate_experience(
                training_agent, reference_agent,
                experience_file,
                num_games=args.games_per_batch,
                board_size=args.board_size,
                num_workers=args.num_workers,
                parallel_games=args.parallel_games,
                gpu_workers=args.gpu_workers)
            train_on_experience(
                training_agent, tmp_agent, experience_file,
                lr=args.lr, batch_size=args.bs)
            shutil.move(tmp_agent, training_agent)
            total_games += args.games_per_batch

        wins = evaluate(
            training_agent, reference_agent,
            num_games=args.eval_games,
            num_workers=args.num_workers,
            board_size=args.board_size,
            parallel_games=args.parallel_games,
            gpu_workers=args.gpu_workers)
        num_eval = args.eval_games
        print('Won %d / %d games (%.3f)' % (
            wins, num_eval, float(wins) / num_eval))
        logf.write('Won %d / %d games (%.3f)\n' % (
            wins, num_eval, float(wins) / num_eval))

        if wins >= 0.55 * num_eval:
            next_filename = os.path.join(
                args.work_dir,
                'zero_agent_v2_%08d.hdf5' % (total_games,))
            shutil.copy(training_agent, next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n' % next_filename)
        else:
            print('Keep learning\n')

        iteration += 1
        if args.elo_games:
            run_elo_matches(args, generations, training_agent, iteration,
                            total_games, logf)

        logf.flush()
        if args.max_iterations and iteration - start_iteration >= args.max_iterations:
            break
    logf.close()


if __name__ == '__main__':
    main()
