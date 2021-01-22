import cityflow
import numpy as np



class Movement:
    """
    The class defining a Movement on an intersection, a Movement of vehicles from incoming road -> outgoing road
    """
    def __init__(self, ID, in_road, out_road, in_lanes, out_lanes, lane_length, phases=[]):
        """
        initialises the Movement
        :param ID: the unique (for a given intersection) ID associated with the movement
        :param in_road: the incoming road of the movement
        :param out_road: the outgoing road of the movement
        :param in_lanes: the incoming lanes of the movement
        :param out_lanes: the outgoing lanes of the movement
        :param lane_length: the length of the incoming lane (if there is more than one incoming lane we assume they have the same length)
        :param phases: the indices of phases for which the give movement is enabled
        """
        self.ID = ID
        
        self.in_road = in_road
        self.out_road = out_road

        self.in_lanes = in_lanes
        self.out_lanes = out_lanes

        self.length = lane_length
        self.phases = phases

        self.arr_vehs_num = []
        self.dep_vehs_num = []

        self.move_type = None #1 -> turn right, 2 -> turn left, 3 -> go straight
        self.max_saturation = 2.2
        self.max_speed = 11
        self.pass_time = int(np.ceil(self.length / self.max_speed))


        self.prev_vehs = set()
        self.waiting_time = 0
        self.max_waiting_time = 0
        self.waiting_time_list = []

        self.arr_rate = 0
        self.green_time = 0
        self.priority = 0

    def update_wait_time(self, action, green_time, waiting_vehs):
        if  [x for x in self.out_lanes if waiting_vehs[x] > 0]:
            self.waiting_time += green_time + 2
        else:
            self.waiting_time_list.append(self.waiting_time)
            if  self.waiting_time > self.max_waiting_time:
                self.max_waiting_time = self.waiting_time
            self.waiting_time = 0
            
    def get_dep_veh_num(self, start_time, end_time):
        return np.sum(self.dep_vehs_num[start_time: end_time])

    def get_arr_veh_num(self, start_time, end_time):
        return np.sum(self.arr_vehs_num[start_time: end_time])

    def update_arr_dep_veh_num(self, lanes_vehs):
        current_vehs = set()

        for lane in self.in_lanes:
            current_vehs.update(lanes_vehs[lane])

        dep_vehs = len(self.prev_vehs - current_vehs)  
        self.dep_vehs_num.append(dep_vehs)
        self.arr_vehs_num.append(len(current_vehs) - (len(self.prev_vehs) - dep_vehs))
        self.prev_vehs = current_vehs

        

class Phase:
    """
    The class defining a Phase on an intersection, a Phase is defined by Movements which are enabled by it (given the green light)
    """
    def __init__(self, ID="", movements=[]):
        """
        initialises the Phase
        :param ID: the unique (for a given intersection) ID associated with the Phase, used by the engine to set phase on
        :param movements: the indeces of Movements which the Phase enables
        """
        self.ID = ID

        self.movements = movements
        self.vector = None
