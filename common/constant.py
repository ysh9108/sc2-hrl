# -*- coding: utf-8 -*-
from enum import Enum


class SubPoliciesName(Enum):

    SCOUT = 0
    ATTACK = 1
    BUILD = 2
    TRAIN = 3

    # BUILD_PYLON = 2
    # BUILD_GATEWAY = 3
    # BUILD_CYBERNETICS_CORE = 4
    # TRAIN_ZEALOT = 5
    # TRAIN_STALKER = 6
    # TRAIN_WORKER = 7


FRAMES_PER_ONE_STEP = 23

N_SCREEN_CHANNELS = 36

N_MINIMAP_CHANNELS = 36

spatial_dim = 84

NUM_OF_POLICIES = 4

Controller_K_Step = 8
EMBEDDING_SIZE = 9

# FEATURE_LIST_MINIMAP = (
#     'height_map',
#     'visibility',
#     'creep',
#     'camera',
#     'player_id',
#     'player_relative',
#     'selected',
#     # TDB - Customized Features
# )
#
# FEATURE_LIST_SCREEN = (
#     'height_map',
#     'visibility',
#     'creep',
#     'power',
#     'player_id',
#     'player_relative',
#     'unit_type',
#     'selected',
#     'hit_points',
#     'energy',
#     'shields',
#     'unit_density',
#     'unit_density_aa',
#     # TDB - Customized Features
# )

SUB_POLICIES_NAME = [
    'SCOUT',
    'ATTACK',
    'BUILD',
    'TRAIN',
    # 'BUILD_PYLON',
    # 'BUILD_GATEWAY',
    # 'BUILD_CYBERNETICS_CORE',
    # 'TRAIN_ZEALOT',
    # 'TRAIN_STALKER',
    # 'TRAIN_WORKER',

    # 가속 : EFFECT_CHRONOBOOSTENERGYCOST
    # WarpGate
    # Research_warpgate
    # gateway -> morph_warpgate
]

SUB_POLICIES_DICT={
    # Micro 관련 policies
    'SCOUT':{'category':'micro','index':0, 'target':['0','1','2','3','4','5','6','7']},
    'ATTACK':{'category':'micro','index':1, 'target':['0','1','2','3','4','5','6','7']},
    # Create 관련 policies
    'BUILD': {'category': 'create', 'index': 2, 'target':['PYLON','GATEWAY','CYBERNETICS_CORE']},
    'TRAIN': {'category': 'create', 'index': 3, 'target':['ZEALOT','STALKER','PROBE']},
    # Create 관련 policies
    # 'BUILD_PYLON':{'category':'create','index':2},
    # 'BUILD_GATEWAY':{'category':'create','index':3},
    # 'BUILD_CYBERNETICS_CORE':{'category':'create','index':4},
    # 'TRAIN_ZEALOT':{'category':'create','index':5},
    # 'TRAIN_STALKER':{'category':'create','index':6},
    # 'TRAIN_WORKER':{'category':'create','index':7},
}
