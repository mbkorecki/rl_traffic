from intersection import Movement, Phase
from fixed_agent import Fixed_Agent
import random

class Random_Agent(Fixed_Agent):

    def __init__(self, eng, ID=''):
        super().__init__(eng, ID)
        self.agents_type = 'random'
    
    def act(self, lanes_count):
        phaseID = random.randint(1, len(self.phases))
        return self.phases[phaseID]
