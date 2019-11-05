import sc2
from sc2.constants import *
from sc2.position import Point2, Point3, Rect
from common.reward import Reward
import sys


class Scout:
    def __init__(self, agent, is_debug=False):
        self.agent = agent
        self.is_debug = is_debug
        self.unit = None
        self.enemy_location = self.agent.enemy_start_locations[0]
        self.unit_discovered = dict()
        self._flag_unit_dead = False

        if self.enemy_location == tuple((28.5, 60.5)):  # left-top
            self.flag_point = [
                Point2((43.0, 61.6)),
                Point2((32.6, 70.7)),
                Point2((19.9, 66.2)),
                Point2((19.5, 55.8)),
                Point2((34.0, 52.9)),
                Point2((42.7, 57.7)),
            ]
            self.enemy_location_range = Rect((19.0, 51.7, 24.1, 19.5))
        else:
            self.flag_point = [
                Point2((46.8, 27.8)),
                Point2((60.1, 34.3)),
                Point2((67.9, 33.1)),
                Point2((69.1, 23.3)),
                Point2((58.1, 18.8)),
                Point2((51.2, 20.7)),
            ]
            self.enemy_location_range = Rect((45.3, 16.6, 24.1, 18.5))

        self.action_sended = False

    async def on_step(self, iteration):
        self._unit = self.unit
        self.unit = self.agent.m_worker.get_scouting_worker_tag()
        if self.unit == -1: return 0

        unit = self.agent.units.find_by_tag(self.unit)
        if unit is None: return 0

        if self._unit != self.unit:
            self.action_sended = False
        self._unit = self.unit

        action_todo = []

        if unit.position.distance_to_point2(self.flag_point[0]) < 0.8:
            self.flag_point.append(self.flag_point.pop(0))
            self.action_sended = False

        if self.action_sended == False:
            action_todo.append(unit.move(self.flag_point[0]))
            self.action_sended = True

        if len(action_todo):
            try:
                await self.agent.do_actions(actions=action_todo)
            except Exception as e:
                if self.is_debug:
                    print('스카웃 on_step 예외 발생 : ', e, ' | ', sys.exc_info())

        if self.is_debug:
            for i, p in enumerate(self.flag_point):
                p_3 = Point3((p.x, p.y, 12))
                if i == 0:
                    self.agent._client.debug_sphere_out(p=p_3, r=2, color=(255, 0, 255))
                else:
                    self.agent._client.debug_sphere_out(p=p_3, r=2, color=(0, 0, 255))

            # Scout tag.agent
            self.agent._client.debug_sphere_out(p=unit.position3d, r=1, color=(0, 0, 255))
            self.agent._client.debug_text_world(text='Scout:' + unit.position.__repr__(), pos=unit.position3d, size=8, color=(255, 255, 255))

            # Enemy-base tag
            enemy_position = Point3((self.agent.enemy_start_locations[0].position.x,
                                     self.agent.enemy_start_locations[0].position.y,
                                     12))
            self.agent._client.debug_sphere_out(p=enemy_position, r=3, color=(0, 0, 255))
            self.agent._client.debug_text_world(text='Enemy_base', pos=enemy_position, size=10, color=(255, 255, 255))

            p_3 = Point3((self.flag_point[0].x, self.flag_point[0].y, 12))
            self.agent._client.debug_line_out(p0=unit.position3d, p1=p_3, color=(255, 255, 255))

            p1 = Point3((self.enemy_location_range.x, self.enemy_location_range.y, 11))
            p2 = Point3((self.enemy_location_range.x + self.enemy_location_range.width, self.enemy_location_range.y + self.enemy_location_range.height, 14))
            self.agent._client.debug_box_out(p_min=p1, p_max=p2, color=(1, 0, 0))

        '''
        Reward
        1. 적군 영역 안에 들어오면  +0.1
        2. 새로운 건물 발견 시      +5
        3. 새로운 유닛 발견 시      +2
        4. 정찰유닛이 죽으면        -10
        '''

        reward = 0
        if unit.is_visible:
            if self.isInRect(unit.position, self.enemy_location_range):
                reward += Reward.SCOUT_IN_ENEMY_AREA  # 1

        for u in self.agent.known_enemy_units:
            if (u.tag in self.unit_discovered.keys()) and (self.unit_discovered[u.tag] - iteration >= 100):
                # 이미 발견한 유닛이고 100 iteration 뒤에 발견된 경우
                self.unit_discovered[u.tag] = iteration

                if u.is_structure:
                    reward += Reward.SCOUT_ENEMY_NEW_BUILDING
                else:
                    reward += Reward.SCOUT_ENEMY_NEW_UNITS
            elif u.tag not in self.unit_discovered.keys():
                # 처음 발견한 유닛일 경우
                self.unit_discovered[u.tag] = iteration

                if u.is_structure:
                    reward += Reward.SCOUT_ENEMY_NEW_BUILDING
                else:
                    reward += Reward.SCOUT_ENEMY_NEW_UNITS

        if self._flag_unit_dead:
            reward += Reward.SCOUT_DEAD
            self._flag_unit_dead = False

        return reward

    def isInRect(self, p: Point2, r: Rect) -> bool:
        '''
        :param p: 유닛의 위치
        :param r: 범위 (Rect)
        :return: 유닛이 특정 범위에 속해있는지 반환
        '''
        if (p.x >= r.x and p.x <= r.x + r.width) and (p.y >= r.y and p.y <= r.y + r.height):
            return True
        else:
            return False

    def on_unit_dead(self) -> None:
        '''
        정찰 유닛이 죽었을 때 호출되는 함수
        :return: None
        '''
        self._flag_unit_dead = True

        if self.enemy_location == tuple((28.5, 60.5)):  # left-top
            self.flag_point = [
                Point2((43.0, 61.6)),
                Point2((32.6, 70.7)),
                Point2((19.9, 66.2)),
                Point2((19.5, 55.8)),
                Point2((34.0, 52.9)),
                Point2((42.7, 57.7)),
            ]
        else:
            self.flag_point = [
                Point2((46.8, 27.8)),
                Point2((60.1, 34.3)),
                Point2((67.9, 33.1)),
                Point2((69.1, 23.3)),
                Point2((58.1, 18.8)),
                Point2((51.2, 20.7)),
            ]
