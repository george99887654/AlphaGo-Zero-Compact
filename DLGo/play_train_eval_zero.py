import argparse
import datetime
import filecmp
import glob
import itertools
import json
import multiprocessing
import os
import random
import re
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


def play_games_continuous(agent1, agent2, next_game, board_size,
                          temperature_fn, slots, collect=False,
                          num_games=None, label=''):
    """Play games between agent1 and agent2, keeping `slots` games in
    progress at once: whenever a game ends, the next one from `next_game()`
    starts in its place, so batches stay full instead of waiting for their
    longest game.

    `next_game()` returns the next global game number, or None when there
    are no more. agent1 plays black in even-numbered games and white in odd
    ones. Each game uses temperature_fn(moves played in that game). Every
    step, each agent searches all games where it is to move together, so
    its network evaluates their positions in batches.

    Returns a list of (game number, GameRecord, agent1 colour, collectors)
    in the order games finished; collectors is (agent1's, agent2's)
    ZeroExperienceCollector, or None if not collecting.
    """
    max_moves = 2 * board_size * board_size
    games, moves, colors, collectors = {}, {}, {}, {}
    finished = []

    def start_games():
        while len(games) < slots:
            g = next_game()
            if g is None:
                return
            games[g] = GameState.new_game(board_size)
            moves[g] = []
            colors[g] = Player.black if g % 2 == 0 else Player.white
            if collect:
                pair = (zero.ZeroExperienceCollector(),
                        zero.ZeroExperienceCollector())
                for c in pair:
                    c.begin_episode()
                collectors[g] = pair

    start_games()
    while games:
        active = list(games)
        # Decide who is to move in each game *before* anyone moves, so every
        # game advances exactly one ply per step.
        agent1_to_move = {g: games[g].next_player == colors[g] for g in active}
        # Both agents search in lock-step, so each search round evaluates
        # the leaves of all games together.
        jobs, game_lists = [], []
        for agent_idx, agent in enumerate((agent1, agent2)):
            idx = [g for g in active if agent1_to_move[g] == (agent_idx == 0)]
            game_lists.append(idx)
            jobs.append((
                agent, [games[g] for g in idx],
                [collectors[g][agent_idx] for g in idx] if collect else None,
                [temperature_fn(len(moves[g])) for g in idx]))
        for idx, chosen in zip(game_lists, zero.select_moves_multi(jobs)):
            for g, move in zip(idx, chosen):
                moves[g].append(move)
                games[g] = games[g].apply_move(move)
        for g in active:
            if games[g].is_over() or len(moves[g]) >= max_moves:
                result = scoring.compute_game_result(games[g])
                record = GameRecord(moves=moves[g], winner=result.winner,
                                    margin=result.winning_margin)
                finished.append((g, record, colors[g],
                                 collectors.get(g) if collect else None))
                print('%sGame %d%s: %d moves, %s, agent %d wins' % (
                    label, g + 1, '/%d' % num_games if num_games else '',
                    len(moves[g]), result,
                    1 if result.winner == colors[g] else 2))
                del games[g], moves[g], colors[g]
                collectors.pop(g, None)
        start_games()
    return finished


def play_games_batched(agent1, agent2, num_games, board_size,
                       temperature_fn, collect=False):
    """Play `num_games` games between agent1 and agent2 all at the same time
    (game numbers 0..num_games-1). Returns (records, agent1_colors,
    collectors) in game order; collectors is None if not collecting."""
    numbers = iter(range(num_games))
    done = play_games_continuous(
        agent1, agent2, lambda: next(numbers, None), board_size,
        temperature_fn, slots=num_games, collect=collect,
        num_games=num_games)
    done.sort(key=lambda item: item[0])
    records = [rec for _, rec, _, _ in done]
    colors = [col for _, _, col, _ in done]
    collectors = [c for _, _, _, c in done] if collect else None
    return records, colors, collectors


def shared_game_source(counter, num_games):
    """next_game() for play_games_continuous that takes one game at a time
    from the counter shared by all workers."""
    def next_game():
        start, count = claim_games(counter, num_games, 1)
        return start if count else None
    return next_game


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


def start_inference_server(model_files, num_workers):
    """Start the GPU inference server process for `num_workers` search
    workers. Returns (server process, per-worker connection tuples)."""
    request_q = multiprocessing.Queue()
    response_qs = [multiprocessing.Queue() for _ in range(num_workers)]
    server = multiprocessing.Process(
        target=zero.run_inference_server,
        args=(list(model_files), request_q, response_qs))
    server.start()
    return server, [(request_q, response_qs[i], i) for i in range(num_workers)]


def stop_inference_server(server):
    """Wait for the server to exit after all workers said 'done'."""
    server.join(timeout=60)
    if server.is_alive():
        server.terminate()
        server.join()


def load_worker_agents(agent1_filename, agent2_filename, remote):
    """Load the two agents for a worker. With `remote` (a connection tuple
    from start_inference_server) the networks are not built locally: the
    agents send their evaluations to the inference server instead
    (model key 0 = agent1, 1 = agent2)."""
    if remote is None:
        return load_agent(agent1_filename), load_agent(agent2_filename)
    request_q, response_q, worker_id = remote
    agents = []
    for key, filename in enumerate((agent1_filename, agent2_filename)):
        with h5py.File(filename, 'r') as h5file:
            agent = zero.load_zero_agent(h5file, load_model=False)
        agent.remote = zero.RemoteInference(request_q, response_q,
                                            worker_id, key)
        agents.append(agent)
    return agents


def worker_label(use_gpu, remote):
    kind = 'search' if remote is not None else ('GPU' if use_gpu else 'CPU')
    return '[%s worker %d] ' % (kind, os.getpid())


def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games,
                 experience_filename,
                 counter,
                 parallel_games=32,
                 use_gpu=True,
                 remote=None,
                 record_agent2=True):
    """Worker: play self-play games in batches of `parallel_games`, taking
    batches from the shared `counter` until `num_games` games have been
    claimed across all workers, and save the experience it collected.
    With `remote`, network evaluations go to the GPU inference server.
    With record_agent2=False only agent 1's (the learner's) moves are
    kept, e.g. in league games against older generations."""
    configure_worker(use_gpu and remote is None)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    try:
        agent1, agent2 = load_worker_agents(
            agent1_filename, agent2_filename, remote)
        label = worker_label(use_gpu, remote)
        print('%sSimulating self-play games, %d at a time...' % (
            label, parallel_games or num_games))
        finished = play_games_continuous(
            agent1, agent2, shared_game_source(counter, num_games),
            board_size, self_play_temperature,
            slots=parallel_games or num_games,
            collect=True, num_games=num_games, label=label)
    finally:
        if remote is not None:
            remote[0].put(('done', remote[2]))

    all_collectors = []
    for _, record, color1, (collector1, collector2) in finished:
        reward1 = 1 if record.winner == color1 else -1
        collector1.complete_episode(reward=reward1)
        collector2.complete_episode(reward=-reward1)
        all_collectors.append(collector1)
        if record_agent2:
            all_collectors.append(collector2)

    all_collectors = [c for c in all_collectors if c.states]
    if not all_collectors:
        return  # this worker got no games; its file stays empty
    experience = zero.combine_experience(all_collectors)
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def require_even_games(num_games, what='match'):
    """Game g gives agent 1 Black when g is even and White when it is odd
    (play_games_continuous), so a match has as many Black as White games
    for each side exactly when it has an even number of games."""
    if num_games % 2:
        raise ValueError('%s needs an even number of games so both agents '
                         'play Black and White equally often (got %d)' % (
                             what, num_games))


def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, board_size, num_workers,
                        parallel_games=32, gpu_workers=1,
                        inference_server=False, record_reference=True):
    require_even_games(num_games, 'self-play')
    experience_files = []
    workers = []
    counter = multiprocessing.Value('i', 0)
    server, remotes = None, [None] * num_workers
    if inference_server:
        server, remotes = start_inference_server(
            (learning_agent, reference_agent), num_workers)
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
                remotes[i],
                record_reference,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()
    if server is not None:
        stop_inference_server(server)
    failed = [w.exitcode for w in workers if w.exitcode != 0]
    if failed:
        raise RuntimeError('%d self-play worker(s) failed' % len(failed))

    # Merge experience buffers (workers that got no games left empty files).
    print('Merging experience buffers...')
    merge_experience_files(experience_files, exp_file)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def merge_experience_files(filenames, exp_file):
    """Combine experience files (empty files are skipped) into exp_file."""
    buffers = []
    for filename in filenames:
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            continue
        with h5py.File(filename, 'r') as expf:
            buffers.append(zero.load_experience(expf))
    combined_buffer = zero.combine_experience(buffers)
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)


def league_pool(work_dir, generations, reference_agent, iteration,
                promoted_count=5, checkpoint_count=3, checkpoints=None):
    """Candidate league opponents for the self-play of `iteration`, as
    generation names: the last `promoted_count` promoted references and the
    checkpoints (best of the first 2^n generations) for the
    `checkpoint_count` largest 2^n < iteration, without duplicates and
    without the current reference (the main self-play games play it).
    `checkpoints`, if given, replaces the 2^n checkpoints (see
    update_curated_checkpoints)."""
    matches = elo.load_matches(os.path.join(work_dir, 'elo_matches.csv'))
    if not matches or not generations:
        return []
    ratings = elo.fit_ratings(matches, anchor=generations[0])
    rows = elo.rating_rows(ratings, matches)
    current, promoted = reference_generations(work_dir, rows,
                                              reference_agent, generations)
    pool = [g for g in promoted if g != current][-promoted_count:]
    sizes = []
    size = checkpoint_window(iteration)
    if checkpoints is not None:
        pool.extend(checkpoints)
        size = 0
    while size >= 1 and len(sizes) < checkpoint_count:
        sizes.append(size)
        size //= 2
    for size in sizes:
        window = generations[:size]
        if window:
            pool.append(max(window,
                            key=lambda g: ratings.get(g, float('-inf'))))
    available = set(generations)
    return [g for g in dict.fromkeys(pool)
            if g != current and g in available]


def checkpoint_candidates(generations, ratings, iteration, block=32,
                          block_start=192):
    """Curated checkpoint candidates: the highest-rated generation of the
    first 2^n generations for every 2^n < iteration, and of every completed
    block of `block` generations from `block_start` on ([192, 223],
    [224, 255], ...), without duplicates."""
    def best(pool):
        return max(pool, key=lambda g: ratings.get(g, float('-inf')))
    candidates = []
    size = 1
    while size < iteration and generations[:size]:
        candidates.append(best(generations[:size]))
        size *= 2
    start = block_start
    while start + block <= len(generations):
        candidates.append(best(generations[start:start + block]))
        start += block
    return list(dict.fromkeys(candidates))


def update_curated_checkpoints(args, generations, iteration, total_games,
                               logf):
    """Bring the curated checkpoint list up to date and return its members.

    Members play each other once, --league-match-games games per pair
    (results go into the Elo history). A member that wins less than
    --league-prune of its games against the strongest member (highest Elo)
    is removed for good. Pairs with the strongest member are played first,
    so hopeless members are dropped before playing anyone else. State
    (pairs played, members removed) is kept in league_checkpoints.json."""
    work_dir = args.work_dir
    state_file = os.path.join(work_dir, 'league_checkpoints.json')
    state = {'pairs': {}, 'removed': {}}
    if os.path.exists(state_file):
        with open(state_file) as f:
            state.update(json.load(f))
    matches_file = os.path.join(work_dir, 'elo_matches.csv')

    def key(a, b):
        return '|'.join(sorted((a, b)))

    def score(a, b):
        """(wins of a, games) in the mutual match of a and b, or None."""
        result = state['pairs'].get(key(a, b))
        if result is None:
            return None
        first, second = sorted((a, b))
        wins = result[0] if a == first else result[1]
        return wins, result[0] + result[1]

    def save():
        with open(state_file + '.tmp', 'w') as f:
            json.dump(state, f, indent=1, sort_keys=True)
        os.replace(state_file + '.tmp', state_file)

    while True:
        ratings = elo.fit_ratings(elo.load_matches(matches_file),
                                  anchor=generations[0])
        members = [g for g in checkpoint_candidates(
            generations, ratings, iteration, args.league_block,
            args.league_block_start) if g not in state['removed']]
        if not members:
            return []
        strongest = max(members, key=lambda g: ratings.get(g, float('-inf')))
        for g in list(members):
            result = score(g, strongest) if g != strongest else None
            if result and result[0] < args.league_prune * result[1]:
                state['removed'][g] = '%d/%d vs %s' % (result[0], result[1],
                                                       strongest)
                members.remove(g)
                logf.write('Curated checkpoints: removed %s (won %d of %d '
                           'against the strongest, %s)\n' % (
                               g, result[0], result[1], strongest))
                save()
        pending = [(strongest, g) for g in members
                   if g != strongest and score(g, strongest) is None]
        if not pending:
            pending = [(a, b) for a, b in itertools.combinations(members, 2)
                       if score(a, b) is None]
        if not pending:
            logf.write('Curated checkpoints for iteration %d: %s (strongest '
                       '%s)\n' % (iteration, ', '.join(members), strongest))
            logf.flush()
            return members
        a, b = pending[0]
        print('Curated checkpoints: %s vs %s (%d games)' % (
            a, b, args.league_match_games))
        wins = evaluate(
            os.path.join(work_dir, a + '.hdf5'),
            os.path.join(work_dir, b + '.hdf5'),
            num_games=args.league_match_games,
            num_workers=args.num_workers, board_size=args.board_size,
            parallel_games=args.parallel_games, gpu_workers=args.gpu_workers,
            inference_server=args.inference_server)
        losses = args.league_match_games - wins
        first, _ = sorted((a, b))
        state['pairs'][key(a, b)] = ([wins, losses] if a == first
                                     else [losses, wins])
        save()
        elo.append_match(matches_file, iteration - 1, total_games, a, b,
                         wins, losses)
        logf.write('Curated checkpoint match %s vs %s: %d-%d\n' % (
            a, b, wins, losses))
        logf.flush()


def self_play_round(args, training_agent, reference_agent, generations,
                    iteration, experience_file, logf, total_games=0):
    """One round of self-play into experience_file. Without a league this
    is --games-per-batch games against the reference. With
    --league-opponents N, N opponents drawn at random from league_pool each
    play --league-games games against the learner (only the learner's moves
    are recorded) and the reference plays the rest."""
    league = []
    if args.league_opponents and generations:
        checkpoints = None
        if args.league_checkpoints == 'curated':
            checkpoints = update_curated_checkpoints(
                args, generations, iteration, total_games, logf)
        pool = league_pool(args.work_dir, generations, reference_agent,
                           iteration, checkpoints=checkpoints)
        league = random.sample(pool, min(args.league_opponents, len(pool)))
        logf.write('League for iteration %d: %s (pool %s)\n' % (
            iteration, ', '.join(league) or 'none', ', '.join(pool)))
        logf.flush()
    main_games = args.games_per_batch - len(league) * args.league_games
    parts = []
    common = dict(board_size=args.board_size, num_workers=args.num_workers,
                  parallel_games=args.parallel_games,
                  gpu_workers=args.gpu_workers,
                  inference_server=args.inference_server)
    if main_games > 0:
        part = experience_file + '.main'
        generate_experience(training_agent, reference_agent, part,
                            num_games=main_games, **common)
        parts.append(part)
    for opponent in league:
        print('League games: learner vs %s (%d games)' % (
            opponent, args.league_games))
        part = experience_file + '.' + opponent
        generate_experience(training_agent,
                            os.path.join(args.work_dir, opponent + '.hdf5'),
                            part, num_games=args.league_games,
                            record_reference=False, **common)
        parts.append(part)
    merge_experience_files(parts, experience_file)
    for part in parts:
        os.unlink(part)


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
                counter, results, parallel_games=32, use_gpu=True,
                remote=None):
    """Worker: play evaluation games (agent1 vs agent2) in batches claimed
    from the shared counter; put (wins, losses) for agent1 on `results`.
    With `remote`, network evaluations go to the GPU inference server."""
    configure_worker(use_gpu and remote is None)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    try:
        agent1, agent2 = load_worker_agents(agent1_fname, agent2_fname,
                                            remote)
        label = worker_label(use_gpu, remote)
        print('%sEvaluating games, %d at a time...' % (
            label, parallel_games or num_games))
        finished = play_games_continuous(
            agent1, agent2, shared_game_source(counter, num_games),
            board_size, eval_temperature,
            slots=parallel_games or num_games,
            num_games=num_games, label=label)
    except BaseException:
        results.put((0, 0))     # keep evaluate() from waiting forever
        raise
    finally:
        if remote is not None:
            remote[0].put(('done', remote[2]))

    wins, losses = 0, 0
    for _, record, color1, _ in finished:
        if record.winner == color1:
            wins += 1
        else:
            losses += 1
    print('%sAgent 1 record (this worker): %d/%d' % (label, wins, wins + losses))
    results.put((wins, losses))


def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size,
             parallel_games=32, gpu_workers=1, inference_server=False):
    require_even_games(num_games, 'evaluation')
    counter = multiprocessing.Value('i', 0)
    results = multiprocessing.Queue()
    workers = []
    server, remotes = None, [None] * num_workers
    if inference_server:
        server, remotes = start_inference_server(
            (learning_agent, reference_agent), num_workers)
    for i in range(num_workers):
        worker = multiprocessing.Process(
            target=eval_worker,
            args=(learning_agent, reference_agent, num_games, board_size,
                  counter, results, parallel_games, i < gpu_workers,
                  remotes[i]))
        worker.start()
        workers.append(worker)
    # Collect results before joining (a process with queued data cannot exit
    # until it is read).
    game_results = [results.get() for _ in workers]
    for worker in workers:
        worker.join()
    if server is not None:
        stop_inference_server(server)
    if any(w.exitcode != 0 for w in workers):
        raise RuntimeError('evaluation worker(s) failed')

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


def generation_of(agent_file, work_dir, generations):
    """Name of the generation snapshot whose file is identical to agent_file
    (promoted references are byte-for-byte copies of a snapshot), or None."""
    if not agent_file or not os.path.exists(agent_file):
        return None
    for gen in reversed(generations):
        path = os.path.join(work_dir, gen + '.hdf5')
        if os.path.exists(path) and filecmp.cmp(agent_file, path, shallow=False):
            return gen
    return None


def reference_generations(work_dir, rows, reference_agent, generations):
    """(current reference generation, promoted generations in order).

    Every promotion saves the learner as zero_agent_v2_<total games>.hdf5,
    a copy of the generation snapshot taken after that many games, so the
    game count in the file name identifies the generation."""
    by_games = {r['total_games']: r['generation'] for r in rows
                if r['generation'] != generations[0]}

    def gen_of(path):
        m = re.match(r'zero_agent_v2_(\d+)\.hdf5$', os.path.basename(path or ''))
        return by_games.get(int(m.group(1))) if m else None

    promoted = [g for g in (gen_of(f) for f in sorted(glob.glob(
        os.path.join(work_dir, 'zero_agent_v2_*.hdf5')))) if g]
    current = gen_of(reference_agent)
    if current is None and reference_agent:
        # The starting reference (nothing promoted yet): usually gen 0.
        current = generation_of(reference_agent, work_dir, generations[:1])
    return current, promoted


def write_elo_report(work_dir, generations, reference_agent, report_dir=None,
                     gate_threshold=0.55, report_name=None):
    """Refit the ratings from the match history and rewrite elo.csv and
    elo_report.html (also copied to report_dir, as rlgo_elo_report.html and
    rlgo_elo.csv, or rlgo_elo_report_<report_name>.html and
    rlgo_elo_<report_name>.csv). Returns the ratings."""
    matches = elo.load_matches(os.path.join(work_dir, 'elo_matches.csv'))
    ratings = elo.fit_ratings(matches, anchor=generations[0])
    rows = elo.write_ratings(os.path.join(work_dir, 'elo.csv'),
                             ratings, matches)
    current, promoted = reference_generations(work_dir, rows,
                                              reference_agent, generations)
    report = os.path.join(work_dir, 'elo_report.html')
    title = 'RLGo training: Elo progress'
    if report_name:
        title += ' (%s)' % report_name
    elo.write_html_report(report, rows, matches, title=title,
                          reference=current, promoted=promoted,
                          gate_threshold=gate_threshold)
    if report_dir:
        suffix = '_' + report_name if report_name else ''
        os.makedirs(report_dir, exist_ok=True)
        shutil.copy(report, os.path.join(
            report_dir, 'rlgo_elo_report%s.html' % suffix))
        shutil.copy(os.path.join(work_dir, 'elo.csv'),
                    os.path.join(report_dir, 'rlgo_elo%s.csv' % suffix))
    return ratings


def checkpoint_window(iteration):
    """Largest power of two 2^n < iteration (0 for iteration 1)."""
    size = 1
    while size * 2 < iteration:
        size *= 2
    return size if size < iteration else 0


def checkpoint_opponent(work_dir, generations, iteration):
    """Elo opponent for the generation of `iteration`: the strongest
    generation (highest current Elo) among the first 2^n generations, for
    the largest n with 2^n < iteration. E.g. iteration 188 plays the best
    of gen_0000..gen_0127. The window doubles as training goes on, so the
    opponent stays a strong, mostly fixed yardstick instead of the random
    gen 0, which every later generation beats 40-0 (a result that says
    nothing about how much stronger it is). Iteration 1 plays gen 0."""
    size = checkpoint_window(iteration)
    pool = generations[:size]
    if not pool:
        return generations[0]
    matches = elo.load_matches(os.path.join(work_dir, 'elo_matches.csv'))
    if not matches:
        return pool[0]
    ratings = elo.fit_ratings(matches, anchor=generations[0])
    return max(pool, key=lambda g: ratings.get(g, float('-inf')))


def run_elo_matches(args, generations, training_agent, iteration,
                    total_games, logf, gating=None, reference_agent=None):
    """Snapshot the learner as a new generation, play it against the best
    of the first 2^n generations (see checkpoint_opponent; generation 0
    with --elo-checkpoint gen0) and the most recent generations, and
    refresh the ratings.

    gating = (reference_file, wins, losses) from this iteration's gating
    match; it is added to the match history when the reference is a known
    generation, since it is a match between the same weights."""
    name = save_generation(args.work_dir, training_agent, iteration)
    if args.elo_checkpoint == 'doubling':
        checkpoint = checkpoint_opponent(args.work_dir, generations, iteration)
        logf.write('Checkpoint opponent for %s: %s (best of the first %d '
                   'generations)\n' % (name, checkpoint,
                                       max(1, checkpoint_window(iteration))))
    else:
        checkpoint = generations[0]
    opponents = [checkpoint] + generations[-args.elo_opponents:]
    opponents = list(dict.fromkeys(opponents))      # dedupe, keep order
    matches_file = os.path.join(args.work_dir, 'elo_matches.csv')
    if gating is not None:
        ref_file, g_wins, g_losses = gating
        ref_gen = generation_of(ref_file, args.work_dir, generations)
        if ref_gen is not None:
            elo.append_match(matches_file, iteration, total_games,
                             name, ref_gen, g_wins, g_losses)
            logf.write('Gating match %s vs %s counted for Elo: %d-%d\n' % (
                name, ref_gen, g_wins, g_losses))
    for opp in opponents:
        wins = evaluate(
            os.path.join(args.work_dir, name + '.hdf5'),
            os.path.join(args.work_dir, opp + '.hdf5'),
            num_games=args.elo_games,
            num_workers=args.num_workers,
            board_size=args.board_size,
            parallel_games=args.parallel_games,
            gpu_workers=args.gpu_workers,
            inference_server=args.inference_server)
        losses = args.elo_games - wins
        elo.append_match(matches_file, iteration, total_games,
                         name, opp, wins, losses)
        logf.write('Elo match %s vs %s: %d-%d\n' % (name, opp, wins, losses))
    generations.append(name)

    ratings = write_elo_report(args.work_dir, generations, reference_agent,
                               args.report_dir, args.gate_threshold,
                               args.report_name)
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
    parser.add_argument('--gate-threshold', type=float, default=0.55,
                        help='fraction of the --eval-games the learner must '
                             'win against the reference to replace it. With '
                             '100 games, 0.55 promotes an equally strong '
                             'learner 18%% of the time, 0.58 only 7%%')
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='stop after this many evaluations (0 = run forever)')
    parser.add_argument('--parallel-games', '-p', type=int, default=32,
                        help='games each worker plays simultaneously; their '
                             'network evaluations are batched (0 = all at once)')
    parser.add_argument('--elo-games', type=int, default=0,
                        help='after each iteration, play this many games '
                             'against a checkpoint generation (see '
                             '--elo-checkpoint) and the previous '
                             '--elo-opponents generations and update '
                             'elo.csv / elo_report.html (0 = off)')
    parser.add_argument('--elo-opponents', type=int, default=2)
    parser.add_argument('--elo-checkpoint', choices=['doubling', 'gen0'],
                        default='doubling',
                        help='doubling: at iteration k play the highest-rated '
                             'of the first 2^n generations (largest 2^n < k); '
                             'gen0: always play generation 0 (gen 0 stays '
                             'the 0-Elo anchor either way)')
    parser.add_argument('--report-dir',
                        help='also copy elo_report.html (as '
                             'rlgo_elo_report.html) and elo.csv here after '
                             'every iteration, e.g. ../schematics')
    parser.add_argument('--report-name',
                        help='name of this run in the report: the copies in '
                             '--report-dir become rlgo_elo_report_<name>.html '
                             'and rlgo_elo_<name>.csv, so runs keep separate '
                             'reports')
    parser.add_argument('--stop-at-iteration', type=int, default=0,
                        help='stop once this many iterations (generations) '
                             'are complete, counting the whole run, so a '
                             'resumed run stops at the same place (0 = never)')
    parser.add_argument('--gpu-workers', type=int, default=1,
                        help='how many of the --num-workers use the GPU; the '
                             'rest run single-threaded on the CPU (ignored '
                             'with --inference-server)')
    parser.add_argument('--league-opponents', type=int, default=0,
                        help='each self-play round, this many opponents drawn '
                             'at random from the league (last 5 promoted '
                             'references and the 3 latest best-of-first-2^n '
                             'checkpoints, excluding the current reference) '
                             'each play --league-games games against the '
                             'learner; only the learner\'s moves are recorded '
                             'and the reference plays the rest of -g '
                             '(0 = off; needs --elo-games)')
    parser.add_argument('--league-games', type=int, default=32)
    parser.add_argument('--league-checkpoints', choices=['latest', 'curated'],
                        default='latest',
                        help='checkpoint part of the league pool. latest: '
                             'best of the first 2^n generations for the 3 '
                             'largest 2^n. curated: best of the first 2^n '
                             'for every 2^n plus the best of each completed '
                             '--league-block from --league-block-start; '
                             'members play each other --league-match-games '
                             'games once and any member winning less than '
                             '--league-prune against the strongest is '
                             'removed (the last 5 promoted are never removed)')
    parser.add_argument('--league-block', type=int, default=32)
    parser.add_argument('--league-block-start', type=int, default=192)
    parser.add_argument('--league-match-games', type=int, default=100)
    parser.add_argument('--league-prune', type=float, default=0.05)
    parser.add_argument('--inference-server', action='store_true',
                        help='one process evaluates the network on the GPU '
                             'for all --num-workers search workers, which '
                             'then do only the tree search on the CPU')
    parser.add_argument('--komi', type=float, default=scoring.DEFAULT_KOMI,
                        help="points added to White's score (default "
                             '%(default)s, the usual 9x9 value; runs before '
                             'v1.1.7 used 1.5, which favoured Black)')

    args = parser.parse_args()
    if not args.resume and not (args.lagent and args.ragent):
        parser.error('--lagent and --ragent are required unless --resume')
    if args.league_opponents and (
            args.league_opponents * args.league_games >= args.games_per_batch):
        parser.error('--league-opponents x --league-games must be less '
                     'than --games-per-batch')
    # Every match alternates colours by game number, so even game counts
    # give each pair of agents equally many Black and White games.
    counts = {'--eval-games': args.eval_games, '--elo-games': args.elo_games,
              '--games-per-batch minus the league games':
                  args.games_per_batch - (args.league_opponents *
                                          args.league_games)}
    if args.league_opponents:
        counts['--league-games'] = args.league_games
        if args.league_checkpoints == 'curated':
            counts['--league-match-games'] = args.league_match_games
    for flag, n in counts.items():
        if n % 2:
            parser.error('%s must be even so both agents play Black and '
                         'White equally often (got %d)' % (flag, n))
    # Workers are new processes that read the komi from the environment.
    os.environ['RLGO_KOMI'] = repr(args.komi)
    scoring.KOMI = args.komi
    os.makedirs(args.work_dir, exist_ok=True)

    logf = open(args.log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Komi %s\n' % args.komi)
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
    if args.stop_at_iteration and iteration >= args.stop_at_iteration:
        print('Iteration %d already reached (--stop-at-iteration %d); '
              'nothing to do' % (iteration, args.stop_at_iteration))
        logf.close()
        return
    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n' % (total_games,))
        for i in range(args.rounds_per_eval):
            print('Round %d/%d' % (i + 1, args.rounds_per_eval))
            self_play_round(args, training_agent, reference_agent,
                            generations, iteration + 1, experience_file,
                            logf, total_games)
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
            gpu_workers=args.gpu_workers,
            inference_server=args.inference_server)
        num_eval = args.eval_games
        print('Won %d / %d games (%.3f)' % (
            wins, num_eval, float(wins) / num_eval))
        logf.write('Won %d / %d games (%.3f)\n' % (
            wins, num_eval, float(wins) / num_eval))

        gating = (reference_agent, wins, num_eval - wins)
        if wins >= args.gate_threshold * num_eval - 1e-9:
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
                            total_games, logf, gating=gating,
                            reference_agent=reference_agent)

        logf.flush()
        if args.max_iterations and iteration - start_iteration >= args.max_iterations:
            break
        if args.stop_at_iteration and iteration >= args.stop_at_iteration:
            logf.write('Stopped at iteration %d (--stop-at-iteration)\n'
                       % iteration)
            print('Reached iteration %d (--stop-at-iteration); stopping'
                  % iteration)
            break
    logf.close()


if __name__ == '__main__':
    main()
