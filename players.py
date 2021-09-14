import itertools
import random
from tkinter.constants import S
import numpy as np
import copy
import cnn
import torch

class CNNPlayer:
    def __init__(self, path, arch="5Conv"):
        if arch=="5Conv":
            self.net = cnn.CNN5Conv()
        elif arch=="3Conv":
            self.net = cnn.CNN3Conv()
        self.net.load(path)
        self.net.eval()
    
    def play(self, place_func, board, me, _):
        board_state = board.getState()
        player_state = (board_state==me)*1
        opponent_state = ((board_state*-1)==me)*1
        net_input = np.concatenate((
            player_state[None, ...], opponent_state[None, ...]), 
            axis=0)

        net_input = torch.Tensor(net_input)
        with torch.no_grad():
            output = self.net(net_input.unsqueeze(0))
        scores, indices = torch.sort(output, descending=True)
        indices = indices[0].numpy()
        
        positions = list(itertools.product(range(8), repeat=2))
        skipped = [(3, 3), (3, 4), (4, 3), (4, 4)]
        positions = [p for p in positions if p not in skipped]
        
        made_move = False

        for i in indices:
            pos = positions[i]
            made_move = place_func(*pos)
            if made_move:
                return True

        return False

class HumanPlayer:
    def play(self, place_func, board, me, _):
        try:
            pos = map(int, map(str.strip, input().split(" ")))
            place_func(*pos)
            return True
        except ValueError:
            return False

class RandomPlayer:
    def play(self, place_func, board, me, _):
        board_size = board.getSize()
        made_move = False
        pos = None
        positions = list(itertools.product(range(board_size), repeat = 2))
        random.shuffle(positions)
        while not made_move and positions:
            pos = positions.pop()
            made_move = place_func(*pos)

        if not made_move and not positions:
            return False

        return True

class PositionalPlayer:
    def __init__(self):
        self.pos_values_8 = np.array([
                                        [100, -20, 10, 5 , 5 , 10, -20, 100],
                                        [-20, -50, -2, -2, -2, -2, -50, -20],
                                        [10 , -2 , -1, -1, -1, -1, -2 , 10 ],
                                        [5  , -2 , -1, -1, -1, -1, -2 , 5  ],
                                        [5  , -2 , -1, -1, -1, -1, -2 , 5  ],
                                        [10 , -2 , -1, -1, -1, -1, -2 , 10 ],
                                        [-20, -50, -2, -2, -2, -2, -50, -20],
                                        [100, -20, 10, 5 , 5 , 10, -20, 100]
                                    ])
        self.pos_values_6 = np.array([
                                        [50 , -20, 5 , 5 , -20, 50 ],
                                        [-20, -50, -2, -2, -50, -20],
                                        [5  , -2 , -1, -1, -2 , 5  ],
                                        [5  , -2 , -1, -1, -2 , 5  ],
                                        [-20, -50, -2, -2, -50, -20],
                                        [50 , -20, 5 , 5 , -20, 50 ],
                                     ])

    def play(self, place_func, board, me, _):
        board_size = board.getSize()
        board_state = board.getState()
        if board_size == 8:
            pos_values = self.pos_values_8
        elif board_size == 6:
            pos_values = self.pos_values_6
        else :
            print("No positional values")
            return False
        #if less than 80 percent of board is filled, use positional evaluation
        endgame = (board.remaining_squares/(board_size**2)) < 0.2
        positions = list(itertools.product(range(board_size), repeat = 2))
        pos_eval = []
        for p in positions:
            tiles_to_flip = board.isValidMove(me, p[0], p[1])
            if not tiles_to_flip:
                continue
            else:
                future_board = copy.deepcopy(board_state)
                future_board[p[0]][p[1]] = me
                for row in tiles_to_flip:
                    future_board[row[0]][row[1]] = me
                if me==-1:
                    future_board = future_board*-1
                if endgame:
                    n_player = sum(sum(future_board==1))
                    n_opponent = sum(sum(future_board==-1))
                    eval = n_player-n_opponent
                else :
                    eval = sum(sum(future_board*pos_values))
                pos_eval.append([p, eval])
        if len(pos_eval)==0:
            return False
        max_val = max(pos_eval, key=lambda x:x[1])[1]
        chosen = random.choice([p[0] for p in pos_eval if p[1]==max_val])
        made_move = place_func(*chosen)
        return True

class MobilityPlayer:
    def __init__(self, w1=0.5, w2=0.5):
        #w1 and w2 balance the importance of number of disks and mobility
        self.w1 = w1
        self.w2 = w2

    def play(self, place_func, board, me, _):
        board_size = board.getSize()
        board_state = board.getState()
        #if less than 80 percent of board is filled, use mobility evaluation
        endgame = (board.remaining_squares/(board_size**2)) < 0.2
        positions = list(itertools.product(range(board_size), repeat = 2))

        pos_eval = []
        for p in positions:
            tiles_to_flip = board.isValidMove(me, p[0], p[1])
            if not tiles_to_flip:
                continue
            else:
                future_board = copy.deepcopy(board_state)
                future_board[p[0]][p[1]] = me
                for row in tiles_to_flip:
                    future_board[row[0]][row[1]] = me
                if me==-1:
                    future_board = future_board*-1
                if endgame:
                    n_player = sum(sum(future_board==1))
                    n_opponent = sum(sum(future_board==-1))
                    eval = n_player-n_opponent
                else :
                    c_player = self.count_corners(future_board, 1)
                    c_opponent = self.count_corners(future_board, -1)
                    mob_player = self.count_mobility(future_board, 1)
                    mob_opponent = self.count_mobility(future_board, -1)
                    if mob_player+mob_opponent==0:
                        mob_ratio = 0
                    else :
                        mob_ratio = ((mob_player-mob_opponent)/(mob_player+mob_opponent))
                    eval = self.w1*(c_player-c_opponent)+self.w2*mob_ratio
                pos_eval.append([p, eval])
        if len(pos_eval)==0:
            return False
        max_val = max(pos_eval, key=lambda x:x[1])[1]
        chosen = random.choice([p[0] for p in pos_eval if p[1]==max_val])
        made_move = place_func(*chosen)
        return True
    
    def count_corners(self, board_state, pid):
        board_size = len(board_state)
        n_corner = 0
        if board_state[0][0] == pid:
            n_corner += 1
        if board_state[0][board_size-1] == pid:
            n_corner += 1
        if board_state[board_size-1][0] == pid:
            n_corner += 1
        if board_state[board_size-1][board_size-1] == pid:
            n_corner += 1
        return n_corner

    def count_mobility(self, board_state, pid):
        board_size = len(board_state)
        positions = list(itertools.product(range(board_size), repeat = 2))
        n = 0
        for p in positions:
            if board_state[p[0]][p[1]] != 0:
                continue
            board_state[p[0]][p[1]] = pid
            otherTile = pid * -1
            # loop through all directions around flipped tile
            for xdirection, ydirection in ((0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)):
                x, y = p[0], p[1]
                x += xdirection # first step in the direction
                y += ydirection # first step in the direction
                if self.isOnBoard(x, y, board_size) and board_state[x][y] == otherTile:
                    # There is a piece belonging to the other player next to our piece.
                    x += xdirection
                    y += ydirection
                    if not self.isOnBoard(x, y, board_size):
                        continue
                    while board_state[x][y] == otherTile:
                        x += xdirection
                        y += ydirection
                        if not self.isOnBoard(x, y, board_size):
                            # break out of while loop, then continue in for loop
                            break
                    if not self.isOnBoard(x, y, board_size):
                        continue
                    if board_state[x][y] == pid:
                        n += 1
                        break
            # restore the empty space
            board_state[p[0]][p[1]] = 0
            
        return n

    def isOnBoard(self, x, y, board_size):
        """
        Returns True if the coordinates are located on the board.
        """
        return x >= 0 and x <= board_size-1 and y >= 0 and y <= board_size-1

def OneHot(index, dim):
    """
    Converts an index into a one-hot encoded column vector.
    """
    a = np.zeros((dim,1))
    a[index] = 1
    return a

if __name__== "__main__":
    from game import Game

    r = RandomPlayer()
    p = PositionalPlayer()
    m = MobilityPlayer()
    h = HumanPlayer()
    c = CNNPlayer("models/test_model.pth")
    g = Game(board_size=8)
    g.addPlayer(r)
    # g.addPlayer(p)
    # g.addPlayer(m)
    # g.addPlayer(h)
    g.addPlayer(c)
    g.run(True)
