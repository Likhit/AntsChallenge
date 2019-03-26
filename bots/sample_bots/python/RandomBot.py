#!/usr/bin/env python
from random import shuffle
from sample_game import *

class RandomBot:
    def do_turn(self, ants, print_to_std_out=True):
        destinations = []
        orders = []
        for a_row, a_col in ants.my_ants():
            # try all directions randomly until one is passable and not occupied
            directions = list(AIM.keys())
            shuffle(directions)
            for direction in directions:
                (n_row, n_col) = ants.destination(a_row, a_col, direction)
                if (not (n_row, n_col) in destinations and
                        ants.passable(n_row, n_col)):
                    order = ants.issue_order((a_row, a_col, direction), print_to_std_out)
                    orders.append(order)
                    destinations.append((n_row, n_col))
                    break
            else:
                destinations.append((a_row, a_col))
        return orders

if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    try:
        Game.run(RandomBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
