from dqn import Agent
from env import Env
import numpy as np
import tensorflow as tf
import os

data_path = '/Users/ScottJeen/OneDrive - University of Cambridge/research/modelling/emerson_data/env'

if __name__ == '__main__':
    env = Env(data_path)
    
