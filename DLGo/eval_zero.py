import argparse
import datetime
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

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


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games,
                 experience_filename,
                 gpu_frac):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename)
    agent2 = load_agent(agent2_filename)

    collector1 = zero.ZeroExperienceCollector()
    collector2 = zero.ZeroExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        collector2.begin_episode()
        agent1.set_collector(collector1)
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
            
        game_record = simulate_game(black_player, white_player, board_size)
        
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
        color1 = color1.other

    experience = zero.combine_experience([collector1, collector2])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, board_size, num_workers):
    experience_files = []
    workers = []
    gpu_frac = 1 / float(num_workers)
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                games_per_worker,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    with h5py.File(first_filename, 'r') as expf:
        combined_buffer = zero.load_experience(expf)
    for filename in other_filenames:
        with h5py.File(filename, 'r') as expf:
            next_buffer = zero.load_experience(expf)
        combined_buffer = zero.combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size):
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

def eval_play_games(args):
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
            
        game_record = eval_simulate_game(black_player, white_player, board_size)
        
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses


def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size):
    games_per_worker = num_games // num_workers
    gpu_frac = 1 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(eval_play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tagent', required=True)
    parser.add_argument('--ragent', required=True)
    parser.add_argument('--games-per-batch', '-g', type=int, default=1000)
    parser.add_argument('--work-dir', '-d')
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--board-size', '-b', type=int, default=19)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--log-file', '-l')

    args = parser.parse_args()

    logf = open(args.log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Starting from %s at %s\n' % (
        args.tagent, datetime.datetime.now()))
    logf.write('Referencing from %s at %s\n' % (
        args.ragent, datetime.datetime.now()))

    learning_agent = args.tagent
    reference_agent = args.ragent
    total_games = 0
    while True:
        print('Reference: %s' % (reference_agent,))
        
        wins = evaluate(
            learning_agent, reference_agent,
            num_games=50,
            num_workers=args.num_workers,
            board_size=args.board_size)
        print('Won %d / 50 games (%.3f)' % (
            wins, float(wins) / 50.0))
        logf.write('Won %d / 50 games (%.3f)\n' % (
            wins, float(wins) / 50.0))
    
        
        logf.flush()


if __name__ == '__main__':
    main()