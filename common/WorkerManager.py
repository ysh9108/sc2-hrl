from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


from collections import deque
from common.constant import SubPoliciesName
from common.reward import Reward

class EnemyUnitManager():

    def __init__(self, agent):
        self.agent = agent


        pass

class UnitManger():

    def __init__(self, agent):
        self.agent = agent

        self.n_nexus = 1
        self.n_pylon = 0
        self.n_gateway = 0
        self.n_assimilator = 0
        self.n_cyberneticscore = 0

        self.n_probe = 12
        self.n_zealot = 0
        self.n_stalker = 0

    def print_units(self):
        print(self.n_nexus)
        print(self.n_pylon)
        print(self.n_gateway)
        print(self.n_assimilator)
        print(self.n_cyberneticscore)
        print(self.n_probe)
        print(self.n_zealot)
        print(self.n_stalker)

    def get_reward(self, died_units):
        total_reward = 0
        reward_nexus = Reward.NEXUS * (self.agent.units(UnitTypeId.NEXUS).owned.amount - self.n_nexus)
        reward_pylon = Reward.PYLON * (self.agent.units(UnitTypeId.PYLON).owned.amount - self.n_pylon)
        reward_gateway = Reward.GATEWAY * (self.agent.units(UnitTypeId.GATEWAY).owned.amount - self.n_gateway)
        reward_assimilator = Reward.ASSIMILATOR * (self.agent.units(UnitTypeId.ASSIMILATOR).owned.amount - self.n_assimilator)
        reward_cyberneticscore = Reward.CYBERNETICSCORE * (self.agent.units(UnitTypeId.CYBERNETICSCORE).owned.amount - self.n_cyberneticscore)

        reward_probe = Reward.PROBE * (self.agent.units(UnitTypeId.PROBE).owned.amount - self.n_probe)
        reward_zealot = Reward.ZEALOT * (self.agent.units(UnitTypeId.ZEALOT).owned.amount - self.n_zealot)
        reward_stalker = Reward.STALKER * (self.agent.units(UnitTypeId.STALKER).owned.amount - self.n_stalker)

        total_reward += reward_nexus
        total_reward += reward_pylon
        total_reward += reward_gateway
        total_reward += reward_assimilator
        total_reward += reward_cyberneticscore
        total_reward += reward_probe
        total_reward += reward_zealot
        total_reward += reward_stalker

        # print('died_units : ', len(died_units))

        # print('# of Zealots : ', self.agent.units(UnitTypeId.ZEALOT).amount)
        # print('# of Stalkers : ', self.agent.units(UnitTypeId.STALKER).amount)

        # # 파괴된 유닛
        # for unit in self.agent.known_enemy_units():
        #     if unit.tag in died_units:
        #         print('destroyed enemy_unit : ', unit)
        if self.agent.selected_sub_policy_id == SubPoliciesName.ATTACK.value:
            reward_attack_units = 0
            n_attack_units = self.agent.units(UnitTypeId.ZEALOT).owned.amount + self.agent.units(UnitTypeId.STALKER).owned.amount
            if n_attack_units < 10:
                reward_attack_units += Reward.ATTACK_PENALTY
            else:
                for unit in self.agent.units(UnitTypeId.ZEALOT):
                    if unit.is_mine:
                        dis = unit.position.distance_to_point2(self.agent.enemy_start_locations[0])
                        reward_attack_units += 1/(dis + Reward.ATTACK_DIS)
                for unit in self.agent.units(UnitTypeId.STALKER):
                    if unit.is_mine:
                        dis = unit.position.distance_to_point2(self.agent.enemy_start_locations[0])
                        reward_attack_units += 1/(dis + Reward.ATTACK_DIS)

            total_reward += reward_attack_units

        return total_reward

    def update_units(self):
        self.n_nexus = self.agent.units(UnitTypeId.NEXUS).owned.amount
        self.n_pylon = self.agent.units(UnitTypeId.PYLON).owned.amount
        self.n_gateway = self.agent.units(UnitTypeId.GATEWAY).owned.amount
        self.n_assimilator = self.agent.units(UnitTypeId.ASSIMILATOR).owned.amount
        self.n_cyberneticscore = self.agent.units(UnitTypeId.CYBERNETICSCORE).owned.amount
        self.n_probe = self.agent.units(UnitTypeId.PROBE).owned.amount
        self.n_zealot = self.agent.units(UnitTypeId.ZEALOT).owned.amount
        self.n_stalker = self.agent.units(UnitTypeId.STALKER).owned.amount

    async def on_step(self, iteration):
        reward = 0

        return reward


class ResourceManager():

    def __init__(self, agent):
        self.agent = agent
        self.assimilators = deque(maxlen=10)
        # self.minerals = []

    def _has_assimilator(self, building):
        for target_building in self.assimilators:
            if target_building['building'].tag == building.tag:
                return True
        return False

    async def on_step(self, iteration):
        for a in self.agent.units(UnitTypeId.ASSIMILATOR):
            if a.is_mine and not self._has_assimilator(a) and a.is_ready:
                self.assimilators.append(dict({'building':a, 'assigned_workers':0}))

class WorkerError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class WorkerManager():

    def __init__(self, agent):
        self.agent = agent
        self.scout_workers = deque(maxlen=1)
        self.mineral_workers = set([])
        self.gas_workers = set([])
        self.build_workers = set([])

        self.all_workers_deque = []
        self.all_workers_deque.append(self.scout_workers)
        self.all_workers_deque.append(self.mineral_workers)
        self.all_workers_deque.append(self.gas_workers)
        self.all_workers_deque.append(self.build_workers)

    def init_unit(self):
        for w in self.agent.units(UnitTypeId.PROBE):
            self.mineral_workers.add(w)

    def _print_tag_all_worker_deque(self):
        # 모든 일꾼 관리자에서 해당 일꾼 태그 프린트
        for target_worker_deque in self.all_workers_deque:
            for worker in target_worker_deque:
                print(worker.tag, '|',end='')
            print('--')

    def _remove_all_worker_deque(self, worker):
        # 모든 일꾼 관리자에서 해당 일꾼을 제거
        for target_worker_deque in self.all_workers_deque:
            for target_worker in target_worker_deque:
                if target_worker.tag == worker.tag:
                    target_worker_deque.remove(target_worker)
                    break

    def _has_all_worker_deque(self, worker):
        for target_worker_deque in self.all_workers_deque:
            for target_worker in target_worker_deque:
                if target_worker.tag == worker.tag:
                    return True
        return False

    def _has_scout_worker_deque(self, worker):
        for target_worker in self.scout_workers:
            if target_worker.tag == worker.tag:
                return True
        return False

    def _has_mineral_worker_deque(self, worker):
        for target_worker in self.mineral_workers:
            if target_worker.tag == worker.tag:
                return True
        return False

    def _has_build_worker_deque(self, worker):
        for target_worker in self.build_workers:
            if target_worker.tag == worker.tag:
                return True
        return False

    def add_scouting_worker(self, worker):
        # TODO : 1개 이상의 정찰 유닛 운영할 경우 수정 필요
        if len(self.scout_workers) == 0:
            self._remove_all_worker_deque(worker)
            # 들어갈 일꾼 관리자에 등록
            self.scout_workers.append(worker)

    def remove_scouting_worker(self, died_worker_tags):
        # TODO : 1개 이상의 정찰 유닛 운영할 경우 수정 필요
        is_removed_unit = False
        if len(self.scout_workers) == 0: return is_removed_unit
        set_dwt = set(died_worker_tags)
        set_sw = set([self.scout_workers[0].tag])
        died_workers = set_dwt & set_sw
        if len(died_workers) > 0:
            self.scout_workers.clear()
            is_removed_unit = True
        return is_removed_unit

    def get_scouting_worker_tag(self):
        if len(self.scout_workers) == 0:
            return -1
        return self.scout_workers[0].tag

    def select_build_worker(self, pos):
        # 일꾼 예외처리
        if len(self.agent.workers) == 0:
            raise WorkerError('일꾼이 존재하지 않습니다.')

        min_dis = 99999
        selected_worker = None
        for worker in self.mineral_workers:
            dis = worker.position.distance_to_point2(pos)
            if min_dis > dis or selected_worker is None:
                min_dis = dis
                selected_worker = worker

        if selected_worker is None:
            selected_worker = self.agent.workers.random
            self._remove_all_worker_deque(selected_worker)
            self.build_workers.add(selected_worker)
            return selected_worker
        else:
            self._remove_all_worker_deque(selected_worker)
            self.build_workers.add(selected_worker)
            return selected_worker

    def _get_closest_mineral_worker(self, pos):
        # 일꾼 예외처리
        if len(self.agent.workers) == 0:
            raise WorkerError('일꾼이 존재하지 않습니다.')

        min_dis = 9999
        selected_worker = None
        for worker in self.mineral_workers:
            dis = worker.position.distance_to_point2(pos)
            if min_dis > dis or selected_worker is None:
                min_dis = dis
                selected_worker = worker

        if selected_worker is None:
            selected_worker = self.agent.workers.random
            self._remove_all_worker_deque(selected_worker)
            return selected_worker
        else:
            self._remove_all_worker_deque(selected_worker)
            return selected_worker

    def update_dead_units(self, dead_units):
        d_units = []
        for w in self.mineral_workers:
            if w.tag in dead_units:
                d_units.append(w)

        for w in d_units:
            self.mineral_workers.remove(w)

        d_units = []
        for w in self.gas_workers:
            if w.tag in dead_units:
                d_units.append(w)

        for w in d_units:
            self.gas_workers.remove(w)

        d_units = []
        for w in self.build_workers:
            if w.tag in dead_units:
                d_units.append(w)

        for w in d_units:
            self.build_workers.remove(w)


    def on_step(self, iteration):
        # print('PROBE amount : ', self.agent.units(UnitTypeId.PROBE).amount)

        todo_actions = []

        # 생산 일꾼 관리
        for worker in self.agent.units(UnitTypeId.PROBE):
            # 어떤 작업 관리자에든 속해있는지 체크
            if worker.is_mine and not self._has_all_worker_deque(worker):
                # 속해있지 않은 일꾼이면 미네랄에 추가
                self.mineral_workers.add(worker)
                # print('MINERAL DEQUE len : ', len(self.mineral_workers))
                # print('-----------', worker.tag, ' | new worker')

        # IDLE 일꾼 관리
        for worker in self.agent.units(UnitTypeId.PROBE):
            # 일꾼이 아무것도 안하며, 정찰도 아니다
            if worker.is_mine and worker.is_idle and not self._has_scout_worker_deque(worker):
                # 일단 모든 작업 관리자에서 제거
                self._remove_all_worker_deque(worker)
                # 미네랄 작업 관리자 개수 체크
                cnt_mineral_workers = len(self.mineral_workers)
                # print('mineral count : ', cnt_mineral_workers, worker.tag, ' | idle worker')
                # 미네랄 작업 관리자로 배정
                self.mineral_workers.add(worker)
                if cnt_mineral_workers != len(self.mineral_workers):
                    mf = self.agent.state.mineral_field.closest_to(self.agent.start_location)
                    # 액션 수행 추가
                    todo_actions.append(worker.gather(mf))
                # else:
                #     print('미네랄에 이미 속해있던 놈')

        # 가스 일꾼 관리
        n_ass = len(self.agent.m_resource.assimilators)
        if len(self.gas_workers) < n_ass * 3:
            n_new_gas_workers = n_ass * 3 - len(self.gas_workers)
            # print('n_new_gas_workers : ', n_new_gas_workers)
            # print('assimilators : ', self.agent.m_resource.assimilators)
            # 미네랄 일꾼들 중 한명을 가져와 가스에 투입
            for ass in self.agent.m_resource.assimilators:
                if ass['assigned_workers'] < 3:
                    target_ass_pos = ass['building'].position
                    try:
                        closest_worker = self._get_closest_mineral_worker(target_ass_pos)
                    except Exception as e:
                        print('가스 일꾼 처리 시 ', e)
                        break
                    self.gas_workers.add(closest_worker)
                    ass['assigned_workers'] += 1
                    todo_actions.append(closest_worker.gather(ass['building']))

        non_orders_workers = []

        for worker in self.build_workers:
            if len(worker.orders) == 0:
                non_orders_workers.append(worker)
            # elif worker.orders[0].ability.id in [AbilityId.HARVEST_GATHER,AbilityId.HARVEST_RETURN]:
            #     non_orders_workers.append(worker)
            # else:
            #     if not worker.orders[0].ability.id in [AbilityId.HARVEST_GATHER,AbilityId.HARVEST_RETURN]:
            #         print('build orders : ', worker.orders[0].ability.id)

        for w in non_orders_workers:
            # print('BEFORE build worker cnt : ', len(self.build_workers))
            self.build_workers.remove(w)
            # print('AFTER build worker cnt : ', len(self.build_workers), ' | build worker tag : ',w.tag)
            # 가까운 미네랄로 추가
            mf = self.agent.state.mineral_field.closest_to(self.agent.start_location)
            # 액션 수행 추가
            todo_actions.append(w.gather(mf))

        #         # 건설 일꾼 관리
        # for worker in self.build_workers:
        #     print('worker orders : ', worker.orders, ' | ', worker.tag)
        #     if len(worker.orders) == 0:
        #         # 일단 모든 작업 관리자에서 제거
        #         self._remove_all_worker_deque(worker)
        #         # 미네랄 작업 관리자 개수 체크
        #         cnt_mineral_workers = len(self.mineral_workers)
        #         print('mineral count : ', cnt_mineral_workers, worker.tag, ' | build worker')
        #         # 미네랄 작업 관리자로 배정
        #         self.mineral_workers.add(worker)
        #         if cnt_mineral_workers != len(self.mineral_workers):
        #             mf = self.agent.state.mineral_field.closest_to(self.agent.start_location)
        #             # 액션 수행 추가
        #             todo_actions.append(worker.gather(mf))
        #             break
        #         else:
        #             print('미네랄에 이미 속해있던 놈')

        # 모아뒀던 액션들 한번에 수행
        if len(todo_actions) > 0:
            # print('@@@@@@@@@@@@')
            # print('self.agnet w : ', len(self.agent.units(UnitTypeId.PROBE)))
            # print('mineral w : ', len(self.mineral_workers), self.mineral_workers)
            # print('gas w : ', len(self.gas_workers), self.gas_workers)
            # print('scout w : ', len(self.scout_workers), self.scout_workers)
            # print('build w : ', len(self.build_workers), self.build_workers)
            # for w in self.build_workers:
            #     print('order : ', w.orders, end=' | ')
            # print('# of Minerals : ', self.agent.minerals)
            # print('# of Vespene : ', self.agent.vespene)
            # print('todo actions 수행', len(todo_actions), ' | is end game : ', self.agent.is_end_game)
            # print('@@@@@@@@@@@@')
            # return self.agent.do_actions(actions=todo_actions)
            return todo_actions