# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from common.constant import SUB_POLICIES_DICT
import tensorflow as tf

import cv2

class ObservationProcessor:
    """
    Observation 처리를 위한 클래스
    Screen과 Minimap의 regularization.
    """

    # N_SCREEN_CHANNELS = 13
    # N_MINIMAP_CHANNELS = 7

    def __init__(self):

        # 스크린 데이터 정규화.
        # 스크린 데이터의 name, scale, type 등을 통해 정규화 과정을 거침.
        # FeatureLayer.get_screen_data()
        # 미니맵 데이터 정규화.
        # 미니맵 데이터의 name, scale, type 등을 통해 정규화 과정을 거침.
        # FeatureLayer.get_minimap_data()
        pass

    def print_obs(self, obs,space=1):
        """
        :param obs:
        KEYS :  dict_keys(['screen', 'minimap'])
        screen : 17
        KEYS :  dict_keys(['height_map', 'visibility_map', 'creep', 'power', 'player_id', 'unit_type', 'selected', 'unit_hit_points', 'unit_hit_points_ratio', 'unit_energy', 'unit_energy_ratio', 'unit_shields', 'unit_shields_ratio', 'player_relative', 'unit_density_aa', 'unit_density', 'effects'])
          height_map : (84, 84, 3)
          visibility_map : (84, 84, 3)
          creep : (84, 84, 3)
          power : (84, 84, 3)
          player_id : (84, 84, 3)
          unit_type : (84, 84, 3)
          selected : (84, 84, 3)
          unit_hit_points : (84, 84, 3)
          unit_hit_points_ratio : (84, 84, 3)
          unit_energy : (84, 84, 3)
          unit_energy_ratio : (84, 84, 3)
          unit_shields : (84, 84, 3)
          unit_shields_ratio : (84, 84, 3)
          player_relative : (84, 84, 3)
          unit_density_aa : (84, 84, 3)
          unit_density : (84, 84, 3)
          effects : (84, 84, 3)
        minimap : 7
        KEYS :  dict_keys(['height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative', 'selected'])
          height_map : (84, 84, 3)
          visibility_map : (84, 84, 3)
          creep : (84, 84, 3)
          camera : (84, 84, 3)
          player_id : (84, 84, 3)
          player_relative : (84, 84, 3)
          selected : (84, 84, 3)
        :return:
        """
        print('KEYS : ', obs.keys())
        for key, value in obs.items():
            if isinstance(value, dict):
                print('{} : {}'.format(key, len(value.items())))
                self.print_obs(value,space+1)
            else:
                print('{}{} : {}'.format(' '*space,key, value.rgb.shape))
                print('value rgb type & value : ',type(value.rgb[0,0,0]),value.rgb[0,0,:])

    def print_preprocObs(self, preprocObs, space=1):
        print('KEYS : ', preprocObs.keys())
        for key, value in preprocObs.items():
            if isinstance(value, dict):
                print('{} : {}'.format(key, len(value.items())))
                self.print_preprocObs(value, space + 1)
            else:
                print('{}{} : {}'.format(' ' * space, key, value.shape))

    def processObs(self, iteration, obs, is_print=False):
        """

        :param iteration:
        :param obs:
        :param is_print:
        :return: dict('screen':[84,84,45],'minimap':[84,84,15],'non_spatial':[])
        """

        # screen에서 제외되는 Features
        # creep : 저그 사용 x
        # selected : 포커스 유닛 이용 x (바로 명령을 내리기 때문)
        # effects : 사이오닉 스톰 ( 당장은 안필요한 것 )
        # minimap에서 제외되는 Features
        # creep : 저그 사용 x


        # spatial data : 공간 정보가 담겨있는 데이터
        # 0~1 사이의 값으로 변경
        # self.preprocObs = {'screen':{}, 'minimap':{},'non_spatial':{}}
        self.preprocObs = {'screen': None, 'minimap': None, 'non_spatial': None}

        # print('one_hot : ', np.array(obs['screen']['unit_type'].one_hot).shape)


        one_hot_controller = np.array(obs['minimap']['unit_type'].one_hot)
        self.preprocObs['minimap'] = one_hot_controller[:,:,0]
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,1]))
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,2]))
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,3]))
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,4]))
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,5]))
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,6]))
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,8])) # 6개
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_controller[:,:,17:])) # 37 - 18 + 1 = 20개 + 6개 = 총 28 channels

        one_hot_minimap_player_relative = np.array(obs['minimap']['player_relative'].one_hot)
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], one_hot_minimap_player_relative)) # + 4 = 총 32 channels

        one_hot_minimap_visibility_map = np.array(obs['minimap']['visibility_map'].one_hot)
        self.preprocObs['minimap'] = np.dstack(
            (self.preprocObs['minimap'], one_hot_minimap_visibility_map))  # + 3 = 총 35 channels

        minimap_structure_busy = np.array(obs['minimap']['structure_busy'].numpy)
        self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], minimap_structure_busy))  # + 3 = 총 36 channels

        one_hot = np.array(obs['screen']['unit_type'].one_hot)
        self.preprocObs['screen'] = one_hot[:,:,0]
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,1]))
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,2]))
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,3]))
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,4]))
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,5]))
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,6]))
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,8])) # 6개
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot[:,:,17:])) # 37 - 18 + 1 = 20개 + 6개 = 총 28 channels

        one_hot_screen_player_relative = np.array(obs['minimap']['player_relative'].one_hot)
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot_screen_player_relative)) # + 4 = 총 32 channels

        one_hot_screen_visibility_map = np.array(obs['minimap']['visibility_map'].one_hot)
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], one_hot_screen_visibility_map)) # + 3 = 총 35 channels

        screen_structure_busy = np.array(obs['screen']['structure_busy'].numpy)
        self.preprocObs['screen'] = np.dstack((self.preprocObs['screen'], screen_structure_busy))  # + 3 = 총 36 channels

        if is_print and iteration == 0:
        # if True:
            self.print_obs(obs)
            print('Minimap shape : ', self.preprocObs['minimap'].shape)
            print('Screen shape : ', self.preprocObs['screen'].shape)

        # for key, value in spatial_screen_data.items():
        #     if self.preprocObs['screen'] is None:
        #         if value[spatial_value_name['Minus_one']]:
        #             self.preprocObs['screen'] = (obs['screen'][key].rgb - 1) / value[spatial_value_name['Max_value']]
        #         else:
        #             self.preprocObs['screen'] = obs['screen'][key].rgb / value[spatial_value_name['Max_value']]
        #     elif value[spatial_value_name['Minus_one']]:
        #         # self.preprocObs['screen'][key] = (obs['screen'][key].rgb - 1) / value[spatial_value_name['Max_value']]
        #         self.preprocObs['screen'] = \
        #             np.dstack((self.preprocObs['screen'], (obs['screen'][key].rgb - 1) / value[spatial_value_name['Max_value']]))
        #     else:
        #         # self.preprocObs['screen'][key] = obs['screen'][key].rgb / value[spatial_value_name['Max_value']]
        #         self.preprocObs['screen'] = \
        #             np.dstack((self.preprocObs['screen'], obs['screen'][key].rgb / value[spatial_value_name['Max_value']]))
        #
        # for key, value in spatial_minimap_data.items():
        #     if self.preprocObs['minimap'] is None:
        #         if value[spatial_value_name['Minus_one']]:
        #             self.preprocObs['minimap'] = (obs['minimap'][key].rgb - 1) / value[spatial_value_name['Max_value']]
        #         else:
        #             self.preprocObs['minimap'] = obs['minimap'][key].rgb / value[spatial_value_name['Max_value']]
        #     elif value[spatial_value_name['Minus_one']]:
        #         # self.preprocObs['minimap'][key] = (obs['minimap'][key].rgb - 1) / value[spatial_value_name['Max_value']]
        #         self.preprocObs['minimap'] = \
        #             np.dstack((self.preprocObs['minimap'], (obs['minimap'][key].rgb - 1) / value[spatial_value_name['Max_value']]))
        #     else:
        #         # self.preprocObs['minimap'][key] = obs['minimap'][key].rgb / value[spatial_value_name['Max_value']]
        #         self.preprocObs['minimap'] = \
        #             np.dstack((self.preprocObs['minimap'],obs['minimap'][key].rgb / value[spatial_value_name['Max_value']]))

        # self.preprocObs['screen']['height_map'] = obs['screen']['height_map'].rgb / 255.0
        # self.preprocObs['screen']['visibility_map'] = obs['screen']['visibility_map'].rgb / 3.0
        # self.preprocObs['screen']['power'] = obs['screen']['power'].rgb
        # self.preprocObs['screen']['player_id'] =(obs['screen']['player_id'].rgb - 1) / 15.0
        # self.preprocObs['screen']['player_relative'] =(obs['screen']['player_relative'].rgb - 1) / 3.0
        # self.preprocObs['screen']['unit_type'] = obs['screen']['unit_type'].rgb / 240.0 # TODO: 확인 필요
        # self.preprocObs['screen']['unit_hit_points'] = obs['screen']['unit_hit_points'].rgb / 255.0
        # self.preprocObs['screen']['unit_hit_points_ratio'] = obs['screen']['unit_hit_points_ratio'].rgb / 255.0
        # self.preprocObs['screen']['unit_energy'] = obs['screen']['unit_energy'].rgb / 255.0
        # self.preprocObs['screen']['unit_energy_ratio'] = obs['screen']['unit_energy_ratio'].rgb / 255.0
        # self.preprocObs['screen']['unit_shields'] = obs['screen']['unit_shields'].rgb / 255.0
        # self.preprocObs['screen']['unit_shields_ratio'] = obs['screen']['unit_shields_ratio'].rgb / 255.0
        # self.preprocObs['screen']['unit_density_aa'] = obs['screen']['unit_density_aa'].rgb / 255.0
        # self.preprocObs['screen']['unit_density'] = obs['screen']['unit_density'].rgb / 16.0
        # self.preprocObs['screen']['effects'] = obs['screen']['effects'].rgb / 255.0
        #
        # self.preprocObs['minimap'] = obs['minimap']['height_map'].rgb / 255.0
        # print('Minimap shape : ', self.preprocObs['minimap'].shape)
        # self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], obs['minimap']['visibility_map'].rgb / 3.0))
        # print('Minimap shape : ', self.preprocObs['minimap'].shape)
        # self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], obs['minimap']['camera'].rgb))
        # print('Minimap shape : ', self.preprocObs['minimap'].shape)
        # self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], (obs['minimap']['player_id'].rgb - 1) / 15.0))
        # print('Minimap shape : ', self.preprocObs['minimap'].shape)
        # self.preprocObs['minimap'] = np.dstack((self.preprocObs['minimap'], (obs['minimap']['player_relative'].rgb - 1) / 3.0))
        # print('Minimap shape : ', self.preprocObs['minimap'].shape)

        # non-spatial data : 공간 정보 외의 데이터
        # 0~1값으로 변경.
        # self.preprocObs['non_spatial']['minearal'] = ...
        # self.preprocObs['non_spatial']['nWorker'] = ...
        # self.preprocObs['non_spatial']['nAttacker'] = ...
        # self.preprocObs['non_spatial']['usedPopulation'] = ...
        # self.preprocObs['non_spatial']['population'] = ...

        return self.preprocObs

spatial_value_name = {
    'Max_value':0,
    'Minus_one':1,
    'Type':2,
    'Shape':3
}

# [맥스 값(나누기 위해), -1을 해줄 건지(기준 0으로), data type, shape]
spatial_screen_data = {
    'height_map' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'visibility_map' : [3.0, False, tf.float32, [None, 84, 84, 3]],
    'power' : [1.0, False, tf.float32, [None, 84, 84, 3]],
    'player_id' : [15.0, False, tf.float32, [None, 84, 84, 3]],
    'player_relative' : [3.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_type' : [240.0, False, tf.float32, [None, 84, 84, 3]], # TODO: 확인 필요
    'unit_hit_points' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_hit_points_ratio' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_energy' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_energy_ratio' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_shields' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_shields_ratio' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_density_aa' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'unit_density' : [16.0, False, tf.float32, [None, 84, 84, 3]],
    'effects' : [255.0, False, tf.float32, [None, 84, 84, 3]]
}

# [맥스 값(나누기 위해), -1을 해줄 건지(기준 0으로), data type, shape]
spatial_minimap_data = {
    'height_map' : [255.0, False, tf.float32, [None, 84, 84, 3]],
    'visibility_map' : [3.0, False, tf.float32, [None, 84, 84, 3]],
    'camera' : [1.0, False, tf.float32, [None, 84, 84, 3]],
    'player_id' : [15.0, False, tf.float32, [None, 84, 84, 3]],
    'player_relative' : [3.0, False, tf.float32, [None, 84, 84, 3]]
}

non_spatial_data = {

}

SCREEN_FEATURE_LIST = (
    # screen
    'height_map',
    'visibility_map',
    'power',
    'player_id',
    'player_relative',
    'unit_type', # TODO: 확인 필요
    'unit_hit_points',
    'unit_hit_points_ratio',
    'unit_energy',
    'unit_energy_ratio',
    'unit_shields',
    'unit_shields_ratio',
    'unit_density_aa',
    'unit_density',
    'effects',
)
MINIMAP_FEATURE_LIST = (
    # minimap
    'height_map',
    'visibility_map',
    'camera',
    'player_id',
    'player_relative',

)

NON_SPATIAL_FEATURE_LIST = (
    # non-spatial
)

AgentScreenInputTuple = namedtuple("AgentInputTuple", SCREEN_FEATURE_LIST)
AgentMinimapInputTuple = namedtuple("AgentMinimapInputTuple", MINIMAP_FEATURE_LIST)
AgentNonSpatialInputTuple = namedtuple("AgentNonSpatialInputTuple", NON_SPATIAL_FEATURE_LIST)

# FEATURE_KEYS = AgentInputTuple(*SCREEN_FEATURE_LIST)
