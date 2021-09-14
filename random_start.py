import numpy as np
import random
from players import *

def select_random():
    states = read_initial_states()
    state = random.choice(states)
    remaining_state = sum(sum(state[0]==0))
    black_score = sum(sum(state[0]==-1))
    white_score = sum(sum(state[0]==1))
    return state[0], remaining_state, black_score, white_score

def read_initial_states(filename='dataset/positions.edax'):
    def to_val(x):
        if x == 'w':
            return 1
        elif x == 'b':
            return -1
        else:
            return 0

    def read_state(line):
        board = np.zeros((8, 8), int)

        for i in range(8):
            for j in range(8):
                board[i,j] = to_val(line[8*(i)+(j)])
        return board, to_val(line[64])

    def read():
        with open(filename, 'r') as f:
            for line in f:
                yield read_state(line)

    return list(read())

if __name__ == "__main__":
    from game import Game

    r = RandomPlayer()
    c = CNNPlayer("models/test_model.pth")
    g = Game(board_size=8, randomize_start=True)
    g.addPlayer(r)
    g.addPlayer(c)
    g.run(True)