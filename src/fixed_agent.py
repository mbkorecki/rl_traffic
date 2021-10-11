from intersection import Movement, Phase
from agent import Agent

class Fixed_Agent(Agent):
    """
    The fixed agent rotating over all possible phases cyclically
    """
    def __init__(self, eng, ID=''):
        """
        initialises the Fixed Agent, which rotates over all possible phases cyclically
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        """
        super().__init__(eng, ID)



    def init_phases(self, eng):
        """
        initialises the phases of the Agent based on the intersection phases extracted from the simulation data
        :param eng: the cityflow simulation engine
        """
        for idx, phase_tuple in enumerate(eng.get_intersection_phases(self.ID)):
            phases = phase_tuple[0]
            types = phase_tuple[1]
            empty_phases = []
            
            new_phase_moves = []
            for move, move_type in zip(phases, types):
                key = tuple(move)
                self.movements[key].move_type = move_type
                new_phase_moves.append(self.movements[key].ID)

            if types and all(x == 1 for x in types): #1 -> turn right
                self.clearing_phase = Phase(idx, new_phase_moves)

            elif new_phase_moves:
                if set(new_phase_moves) not in [set(x.movements) for x in self.phases.values()]:
                    new_phase = Phase(idx, new_phase_moves)                    
                    self.phases.update({idx : new_phase})
            else:
                empty_phases.append(idx)

            if empty_phases:
                self.clearing_phase = Phase(empty_phases[0], [])
                self.phases.update({empty_phases[0] : self.clearing_phase})

        self.phase = self.clearing_phase
        temp_moves = dict(self.movements)
        self.movements.clear()
        for move in temp_moves.values():
            move.phases = []
            self.movements.update({move.ID : move})
            
        for phase in self.phases.values():
            for move in phase.movements:
                if phase.ID not in self.movements[move].phases:
                    self.movements[move].phases.append(phase.ID)
    

    def act(self, lanes_count):
        """
        selects the next phase
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        phaseID = ((self.phase.ID + 1) % len(self.phases))
        keys = [x for x in self.phases.keys()]
        return self.phases[keys[phaseID]]
      
