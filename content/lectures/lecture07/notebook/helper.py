from time import sleep

import numpy as np
from IPython.display import clear_output


class Sprinter:

    def __init__(self, base_speed, variance):
        self.base_speed = base_speed
        self.variance = variance

    @property
    def time(self):
        return np.random.normal(loc=self.base_speed, scale=self.variance)


def run_sim(race, winner):
    for i in range(1, 11):
        clear_output(wait=True)
        print("|START|" + "\n|START|".join(['----' * min(10, int((15 * i) / race[runner])) + '    ' * (
                    10 - min(10, int((15 * i) / race[runner]))) + '|' + runner for runner in race.keys()]))
        sleep(0.5)

    print(f'\nThe winner is {winner[0]} with a time of {winner[1]:.2f}s!')
