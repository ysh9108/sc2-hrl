from multiprocessing import Pipe, Process

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.client import FeatureSetting
from sc2.constants import *
from sc2.position import Point3

import time
import cv2
import numpy as np
from scout import Scout

class Agent(sc2.BotAI):


    def __init__(self, is_render=False):
        super(Agent, self).__init__(is_render=is_render)

    def on_start(self):
        self.scout_unit = None
        self.SCRIPT_SCOUT = Scout(self)

    async def on_step(self, iteration):


        if self.scout_unit is None and self.units(UnitTypeId.PROBE).amount > 0:
            self.scout_unit = self.units(UnitTypeId.PROBE).random

        if self.units.find_by_tag(self.scout_unit.tag) is not None:


            unit = self.units.by_tag(self.scout_unit.tag)

            # Run Script
            await self.SCRIPT_SCOUT.run(unit)

            for i, p in enumerate(self.SCRIPT_SCOUT.flag_point):
                p_3 = Point3((p.x, p.y,12))
                if i == 0:
                    self._client.debug_sphere_out(p=p_3, r=2, color=(255, 0, 255))
                else:
                    self._client.debug_sphere_out(p=p_3, r=2, color=(0, 0, 255))

            # Scout tag
            self._client.debug_sphere_out(p=unit.position3d, r=1, color=(0,0,255))
            self._client.debug_text_world(text='Scout:'+unit.position.__repr__(),pos=unit.position3d, size=8, color=(255,255,255))

            # Enemy-base tag
            enemy_position = Point3((self.enemy_start_locations[0].position.x,
                                    self.enemy_start_locations[0].position.y,
                                    12))
            self._client.debug_sphere_out(p=enemy_position, r=3, color=(0,0,255))
            self._client.debug_text_world(text='Enemy_base',pos=enemy_position, size=10, color=(255,255,255))


            p_3 = Point3((self.SCRIPT_SCOUT.flag_point[0].x, self.SCRIPT_SCOUT.flag_point[0].y, 12))
            self._client.debug_line_out(p0=unit.position3d, p1=p_3, color=(255, 255, 255))

            await self._client.send_debug()
        pass

    def on_end(self, result):

        pass


class SC2Env(Process):
    def __init__(self, idx, is_render=False):
        super(SC2Env, self).__init__()
        self.idx = idx
        self.is_render=is_render


    def run(self):
        super(SC2Env, self).run()

        sc2.run_game(sc2.maps.get("Simple64"), [
            Bot(Race.Protoss, Agent(is_render=self.is_render)),
            Computer(Race.Protoss, Difficulty.Easy)
        ],
            feature_setting=FeatureSetting(
                screen=(84, 84),
                minimap=(84, 84)
            ),
            realtime=False
        )


if __name__ == '__main__':
    num_worker = 1                  # 쓰레드 수 설정
    is_render = False               # GUI 띄울 지 설정

    workers = []
    parent_conns = []

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        worker = SC2Env(idx, is_render=is_render)
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)

