import cityflow
import numpy as np
import random

from intersection import Movement, Phase
from agent import Agent

class ES_Agent(Agent):
    """
    An agent using evolutionary strategies to evolve parameters of the policy network
    """
    def __init__(self, eng, ID=''):
