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
        :param phases: the phases for which the give movement is enabled
        """
        self.ID = ID
        
        self.in_road = in_road
        self.out_road = out_road

        self.in_lanes = in_lanes
        self.out_lanes = out_lanes

        self.length = lane_length
        self.phases = phases

        self.arr_veh_num = []
        self.dep_veh_num = []

        self.max_saturation = 2.2
        self.max_speed = 11
        self.pass_time = int(np.ceil(self.length / self.max_speed))



class Phase:
    """
    The class defining a Phase on an intersection, a Phase is defined by Movements which are enabled by it (given the green light)
    """
    def __init__(self, ID):
        """
        initialises the Phase
        :param ID: the unique (for a given intersection) ID associated with the Phase
        :param movements: the Movements which the Phase enables
        """
        self.ID = ID
