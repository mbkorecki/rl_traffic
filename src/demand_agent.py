import operator
from intersection import Movement, Phase
from agent import Agent

class Demand_Agent(Agent):
    """
    The class defining an agent which controls the traffic lights using the demand based approach
    always prioritizing the phase with the biggest demand
    """
    def __init__(self, eng, ID=''):
        super().__init__(eng, ID)


    def act(self, lanes_count):
        """
        selects phase with biggest demand
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        phases_priority = {}
        for phase in self.phases.values():
            priority = 0
            for moveID in phase.movements:
                priority += self.movements[moveID].get_demand(lanes_count)

            phases_priority.update({phase.ID : priority})

        return self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
      
