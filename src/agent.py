from intersection import Movement, Phase

class Agent:
    """
    The base clase of an Agent, Learning and Analytical agents derive from it, basically defines methods used by both types of agents
    """
    def __init__(self, ID):
        """
        initialises the Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        """
        self.ID = ID
        
        self.movements = {}
        self.phases = {}
        self.clearing_phase = None

        self.action = 0
        self.phase = Phase(ID="")

        self.action_freq = 10
        self.action_type = "act"
        self.clearing_time = 5

    def init_movements(self, eng):
        """
        initialises the movements of the Agent based on the lane links extracted from the simulation roadnet
        the eng.get_intersection_lane_links used in the method takes the intersection ID and returns
        a tuple containing the (in_road, out_road) pair as the first element and
        (in_lanes, out_lanes) as the second element
        :param eng: the cityflow simulation engine
        """
        for idx, roadlink in enumerate(eng.get_intersection_lane_links(self.ID)):
            lanes = roadlink[1][:]
            in_road = roadlink[0][0]
            out_road = roadlink[0][1]
            in_lanes = tuple(set([x[0] for x in lanes]))
            out_lanes = [x[1] for x in lanes]

            for _, length in eng.get_road_lanes_length(in_road):
                lane_length = length
                
            new_movement = Movement(idx, in_road, out_road, in_lanes, out_lanes, lane_length, self.clearing_time)
            self.movements.update({roadlink[0] : new_movement})
            
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

            if new_phase_moves:
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

    def update_arr_dep_veh_num(self, lanes_vehs):
        """
        Updates the list containing the number vehicles that arrived and departed
        :param lanes_vehs: a dictionary with lane ids as keys and number of vehicles as values
        """
        for movement in self.movements.values():
            movement.update_arr_dep_veh_num(lanes_vehs)

            
    def update_wait_time(self, action, green_time, waiting_vehs):
        """
        Updates movements' waiting time - the time a given movement has waited to be enabled
        :param action: the phase to be chosen by the intersection
        :param green_time: the green time the action is going to be enabled for
        :param waiting_vehs: a dictionary with lane ids as keys and number of waiting cars as values
        """
        for movement in self.movements.values():
            movement.update_wait_time(action, green_time, waiting_vehs)
            
    def reset_movements(self):
        """
        Resets the set containing the vehicle ids for each movement and the arr/dep vehicles numbers as well as the waiting times
        the set represents the vehicles waiting on incoming lanes of the movement
        """
        self.phase = self.clearing_phase
        for move in self.movements.values():
            move.prev_vehs = set()
            move.arr_vehs_num = []
            move.dep_vehs_num = []
            move.waiting_time = 0
            move.max_waiting_time = 0
            move.waiting_time_list = []
