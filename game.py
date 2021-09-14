import numpy as np

import board


class Game:
    def __init__(self, board_size, randomize_start=False):
        self.players = []
        self.randomize_start = randomize_start
        self.board = board.Board(board_size=board_size, randomize_start=randomize_start)

    def addPlayer(self, player, log_move_history = True):
        self.players.append((player, log_move_history))

    def getScore(self):
        return self.board.getScore()

    def run(self, show_board = False):
        n_passed = 0
        # Run until both players have passed
        while n_passed < 2:
            n_passed = 0
            for i, player in enumerate(self.players):
                # Pass the player a function it can use to make a move
                # i*2-1 rescales index as if it were an interval
                # (0, 1) -> (-1, 1)
                # Player id
                pid = i*2-1
                if self.randomize_start:
                    pid = pid * -1
                did_move = player[0].play(lambda r,c: self.board.updateBoard(pid,r,c, show_board),
                                   self.board, pid, player[1])

                if show_board:
                    self.board.printBoard()

                if not did_move:
                    n_passed += 1
