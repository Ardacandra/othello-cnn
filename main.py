#!/usr/bin/env python3
#sumber referensi : https://github.com/jpypi/othello-rl
from pprint import pprint

import numpy as np

from game import Game
from players import *
import random

model_path = "models/5conv-3fc-bn/5conv-3fc-bn-epoch-19.pth"
player = CNNPlayer(model_path, arch="5Conv")
# o = RandomPlayer()
# opponent = "r"
o = PositionalPlayer()
opponent = "p"
# o = MobilityPlayer()
# opponent = "m"

match_size = 100

player_wins = 0

for _ in range(match_size):
    # Initialize a new game
    g = Game(8, randomize_start=True)
    
    # Randomise the player order
    random_number = random.random()
    if random_number > 0.5:
        g.addPlayer(player, False)
        g.addPlayer(o, False)
        player_order = 1
    else :
        g.addPlayer(o, False)
        g.addPlayer(player, False)
        player_order = 0
    g.run()

    final_score = list(g.getScore().items())
    final_score.sort()
    ttl = sum(map(lambda x: x[1], final_score))
    #player_score = int(final_score[0][1]/ttl >= 0.5)
    player_score =  (final_score[player_order][1]/ttl - 0.5)*2
    player_wins += player_score > 0

print(player_wins)
# print(sum(player_wins))
# with open("%d-%d-%s.csv"%(n_epochs, match_size, suffix), "w") as f:
    # f.write("\n".join(map(str, player_wins)))
