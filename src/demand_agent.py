import operator
from intersection import Movement, Phase
from agent import Agent

class Demand_Agent(Agent):

    def __init__(self, eng, ID=''):
        super().__init__(ID)

        self.init_movements(eng)
        self.init_phases(eng)

    def act(self, lanes_count):
        phases_priority = {}
        for phase in self.phases.values():
            priority = 0
            for moveID in phase.movements:
                priority += self.movements[moveID].get_demand(lanes_count)

            phases_priority.update({phase.ID : priority})

        return self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
      
