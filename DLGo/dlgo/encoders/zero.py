import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard_fast import Move
from dlgo.gotypes import Player, Point


class ZeroEncoder(Encoder):
    def __init__(self, board_size):
        self.board_size, _ = board_size
        # 0 - 3. our stones with 1, 2, 3, 4+ liberties
        # 4 - 7. opponent stones with 1, 2, 3, 4+ liberties
        # 8. 1 if we get komi
        # 9. 1 if opponent gets komi
        # 10. move would be illegal due to ko
        self.num_planes = 11
# HHY        
    def name(self):
        return 'zero'
# HHY
    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        if game_state.next_player == Player.white:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        board = game_state.board
        grid = board._grid
        neighbor_table = board.neighbor_table
        for r, c, p in self._points():
            go_string = grid.get(p)

            if go_string is None:
                # A ko violation requires a capture (an enemy neighbour
                # string in atari, as in Board.will_capture), so only build a
                # Move and run the full ko check where one is possible.
                captures = False
                for neighbor in neighbor_table[p]:
                    s = grid.get(neighbor)
                    if s is not None and s.color != next_player and \
                            len(s.liberties) == 1:
                        captures = True
                        break
                if captures and game_state.does_move_violate_ko(
                        next_player, Move.play(p)):
                    board_tensor[10][r][c] = 1
            else:
                liberty_plane = min(4, go_string.num_liberties) - 1
                if go_string.color != next_player:
                    liberty_plane += 4
                board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def _points(self):
        """(row index, col index, Point) for every board point, built once."""
        points = getattr(self, '_point_list', None)
        if points is None:
            points = [(r, c, Point(row=r + 1, col=c + 1))
                      for r in range(self.board_size)
                      for c in range(self.board_size)]
            self._point_list = points
        return points

# tag::encode_move[]
    def encode_move(self, move):
        if move.is_play:
            return (self.board_size * (move.point.row - 1) +   # <1>
                (move.point.col - 1))                          # <1>
        elif move.is_pass:
            return self.board_size * self.board_size           # <2>
        raise ValueError('Cannot encode resign move')          # <3>

    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self):
        return self.board_size * self.board_size + 1
# end::encode_move[]

    def shape(self):
        return self.num_planes, self.board_size, self.board_size

# HHY    
def create(board_size):
    return ZeroEncoder(board_size)
# HHY