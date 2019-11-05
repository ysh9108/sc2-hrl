import os
import sys
import time
import random
import datetime
import logging
from collections import deque
from multiprocessing import Process, Pipe, Queue

import numpy as np
import tensorflow as tf

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.client import FeatureSetting
from sc2.position import Point2
from sc2.constants import *
from sc2.data import ActionResult, Result

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.map_position_manager import Map_Position_Manager
from common.scout import Scout
from common.WorkerManager import WorkerManager, ResourceManager, UnitManger
from common.constant import SUB_POLICIES_NAME, Controller_K_Step, SubPoliciesName, NUM_OF_POLICIES, N_MINIMAP_CHANNELS
from common.preprocess import ObservationProcessor

from Controller.discrete_policy import PPO

is_train = True
is_debug = False
is_camera_move = False
is_save_win_game_replay = False
load_model = False

# Game Settings
# 쓰레드 수 설정
num_worker = 16
map_name = "Simple64"
screen_size_px = 84
minimap_size_px = 84
game_time_limit = 1000
my_race = Race.Protoss
enemy_race = Race.Protoss
# difficulty = Difficulty.Easy
enemy_difficulties = [Difficulty.VeryEasy, Difficulty.Easy, Difficulty.Medium, Difficulty.MediumHard, Difficulty.Hard,
                      Difficulty.Harder, Difficulty.VeryHard]
# GUI 띄울 지 설정
is_render = False
realtime = False
# main.py의 play_game_ai의 반복 횟수
n_games_per_one_epoch = 1

# PPO Model Parameters
batch_size = 64
hidden_layer_size = 100
actor_update_steps = 1
critic_update_steps = 1
actor_learning_rate = 0.0000007
critic_learning_rate = 0.0000008
kl_penalty_target = 0.01
kl_penalty_lam = 0.5
clip_epsilon = 0.2
gamma = 0.9
epsilon_max = 1000
epsilon_len = 200
# PPO train_update 호출 횟수
n_learning = 1

# Controller Parameters
state_size = [None, 84, 84, N_MINIMAP_CHANNELS]
action_size = 4
mem_maxlen = 10000

# 저장 및 출력
summary_path = './Summary/Controller/'
save_path = './Model/Controller_'
save_replay_path = '../replays/'
print_interval = 1
save_interval = 5

n_game_epochs = 50000


class Controller:

    def __init__(self,
                 model_name,
                 sess: tf.Session,
                 policy=PPO):

        # 하이퍼 파라미터
        self.batch_size = batch_size
        self.mem_maxlen = mem_maxlen

        # 게임 환경
        self.state_size = state_size
        self.action_size = action_size

        # 세션 생성
        self.sess = sess

        # 모델 생성
        self.global_step = tf.Variable(0, trainable=False)
        self.policy = policy
        self.model = self.policy(global_step=self.global_step,
                                 batch_size=batch_size,
                                 state_size=state_size,
                                 action_size=action_size,
                                 hidden_layer_size=hidden_layer_size,
                                 actor_update_steps=actor_update_steps,
                                 critic_update_steps=critic_update_steps,
                                 actor_learning_rate=actor_learning_rate,
                                 critic_learning_rate=critic_learning_rate,
                                 kl_penalty_target=kl_penalty_target,
                                 kl_penalty_lam=kl_penalty_lam,
                                 clip_epsilon=clip_epsilon,
                                 gamma=gamma,
                                 epsilon_max=epsilon_max,
                                 epsilon_len=epsilon_len,
                                 model_name=model_name)

        # 메모리 생성
        self.memory = deque(maxlen=mem_maxlen)

        # 게임 상태 저장
        self.summary_path = summary_path
        self.save_path = save_path + '0/'
        self.save_format = "actor_lr_{}_critic_lr_{}".format(actor_learning_rate, critic_learning_rate)

        # Saver 생성
        self.saver = tf.train.Saver(max_to_keep=5)
        self.summary, self.merge = self.make_summary()

        # if not (os.path.isdir(save_path + self.save_format)):
        #     os.makedirs(os.path.join(save_path + self.save_format))

        if not (os.path.isdir(summary_path)):
            os.makedirs(os.path.join(summary_path))

        # 모델 로드 결정
        self.load_model = load_model
        if self.load_model:
            ckpt = tf.train.get_checkpoint_state(self.save_path + self.save_format)
            if ckpt is not None:
                print('Model Load : ', ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # 디버그용 프린트 여부 체크
        self.is_print = is_debug

    def make_summary(self):
        with tf.name_scope(name='Loss'):
            self.summary_aloss = tf.placeholder(dtype=tf.float32)
            self.summary_closs = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar("aloss", self.summary_aloss)
            tf.summary.scalar("closs", self.summary_closs)
        with tf.name_scope(name='Rewards_and_Win_Rate'):
            self.summary_steps_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_reward = tf.placeholder(dtype=tf.float32)
            self.summary_win_rate = tf.placeholder(dtype=tf.float32)
            self.summary_game_level = tf.placeholder(dtype=tf.int32)
            tf.summary.scalar("STEPS", self.summary_steps_per_game)
            tf.summary.scalar("reward", self.summary_reward)
            tf.summary.scalar("win_rate", self.summary_win_rate)
            tf.summary.scalar("game_level", self.summary_game_level)
        with tf.name_scope(name='Selected_Sub_Policies'):
            self.summary_scout = tf.placeholder(dtype=tf.float32)
            self.summary_attack = tf.placeholder(dtype=tf.float32)
            self.summary_build = tf.placeholder(dtype=tf.float32)
            self.summary_train = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar("SCOUT", self.summary_scout)
            tf.summary.scalar("ATTACK", self.summary_attack)
            tf.summary.scalar("BUILD", self.summary_build)
            tf.summary.scalar("TRAIN", self.summary_train)
        with tf.name_scope(name='Memory'):
            self.summary_memory_size = tf.placeholder(dtype=tf.int32)
            tf.summary.scalar('Memory_Size', self.summary_memory_size)
        with tf.name_scope(name='Game_info_Resource'):
            self.summary_minerals_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_vespene_per_game = tf.placeholder(dtype=tf.int32)
            tf.summary.scalar("MINERALS", self.summary_minerals_per_game)
            tf.summary.scalar("VESPENE", self.summary_vespene_per_game)
            with tf.name_scope(name='Supply'):
                self.summary_supply_cap_per_game = tf.placeholder(dtype=tf.int32)
                self.summary_supply_left_per_game = tf.placeholder(dtype=tf.int32)
                self.summary_supply_used_per_game = tf.placeholder(dtype=tf.int32)
                tf.summary.scalar("SUPPLY_CAP", self.summary_supply_cap_per_game)
                tf.summary.scalar("SUPPLY_LEFT", self.summary_supply_left_per_game)
                tf.summary.scalar("SUPPLY_USED", self.summary_supply_used_per_game)
        with tf.name_scope(name='Game_info_Buildings'):
            self.summary_nexus_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_pylon_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_gateway_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_cyberneticscore_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_assimilator_per_game = tf.placeholder(dtype=tf.int32)
            tf.summary.scalar("NEXUS", self.summary_nexus_per_game)
            tf.summary.scalar("PYLON", self.summary_pylon_per_game)
            tf.summary.scalar("GATE_WAY", self.summary_gateway_per_game)
            tf.summary.scalar("CYBERNETICSCORE", self.summary_cyberneticscore_per_game)
            tf.summary.scalar("ASSILILATOR", self.summary_assimilator_per_game)
        with tf.name_scope(name='Game_info_Units'):
            self.summary_probe_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_zealot_per_game = tf.placeholder(dtype=tf.int32)
            self.summary_stalker_per_game = tf.placeholder(dtype=tf.int32)
            tf.summary.scalar("PROBE", self.summary_probe_per_game)
            tf.summary.scalar("ZEALOT", self.summary_zealot_per_game)
            tf.summary.scalar("STALKER", self.summary_stalker_per_game)
        return tf.summary.FileWriter(logdir=self.summary_path, graph=self.sess.graph), tf.summary.merge_all()

    def write_summray(self, reward, aloss, closs, win_rate, game_level, actions, memory_size, game_info, episode):
        self.summary.add_summary(
            self.sess.run(self.merge,
                          feed_dict={
                              self.summary_aloss: aloss,
                              self.summary_closs: closs,
                              self.summary_reward: reward,
                              self.summary_win_rate: win_rate,
                              self.summary_game_level: game_level,
                              self.summary_scout: actions[int(SubPoliciesName.SCOUT.value)],
                              self.summary_attack: actions[int(SubPoliciesName.ATTACK.value)],
                              self.summary_build: actions[int(SubPoliciesName.BUILD.value)],
                              self.summary_train: actions[int(SubPoliciesName.TRAIN.value)],
                              self.summary_memory_size: memory_size,
                              self.summary_steps_per_game: game_info['steps'],
                              self.summary_minerals_per_game: game_info['minerals'],
                              self.summary_vespene_per_game: game_info['vespene'],
                              self.summary_nexus_per_game: game_info['nexus'],
                              self.summary_pylon_per_game: game_info['pylon'],
                              self.summary_gateway_per_game: game_info['gateway'],
                              self.summary_cyberneticscore_per_game: game_info['cyberneticscore'],
                              self.summary_assimilator_per_game: game_info['assimilator'],
                              self.summary_probe_per_game: game_info['probe'],
                              self.summary_zealot_per_game: game_info['zealot'],
                              self.summary_stalker_per_game: game_info['stalker'],
                              self.summary_supply_cap_per_game: game_info['supply_cap'],
                              self.summary_supply_left_per_game: game_info['supply_left'],
                              self.summary_supply_used_per_game: game_info['supply_used']
                          }
                          ), episode
        )
        self.summary.flush()

    def save_model(self, save_path_dir):
        if not (os.path.isdir(save_path_dir + self.save_format)):
            os.makedirs(os.path.join(save_path_dir + self.save_format))

        self.saver.save(self.sess, save_path_dir + self.save_format + "/model.ckpt", global_step=self.global_step)

    def append_sample(self, state, action, reward, next_state, info, done):
        self.memory.append((state[0], action, reward, next_state[0], info, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros(shape=[self.batch_size, self.state_size[1], self.state_size[2], self.state_size[3]])
        next_states = np.zeros(shape=[self.batch_size, self.state_size[1], self.state_size[2], self.state_size[3]])
        actions = []
        rewards = []
        infoes = []
        dones = []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            infoes.append(mini_batch[i][4])
            dones.append(mini_batch[i][5])

        if self.is_print:
            # if True:
            print("batch_shape:", np.shape(mini_batch))
            print("states_shape:", np.shape(states))
            print('actions : ', actions)
            print("actions_shape:", np.shape(actions))
            print("rewards_shape:", np.shape(rewards))
            print("next_states_shape:", np.shape(next_states))
            print("info_shape:", np.shape(infoes))
            print("donse_shape:", np.shape(dones))

        aloss, closs = self.update(states, actions, infoes, rewards)
        return aloss, closs

    def choose_action(self, s, info_s):
        # Shape = [1, width, height, n_features]
        s = np.reshape(s, [1, self.state_size[1], self.state_size[2], self.state_size[3]])

        linfo_s = len(info_s)

        info_s = np.reshape(info_s, [1, linfo_s])
        # 액션 추출
        a = self.sess.run(self.model.sample_op, {self.model.tfs: s, self.model.infotfs: info_s})[0]
        # print('[choose action] a : ', action)
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.model.v, {self.model.tfs: s})[0, 0]

    def update(self, s, a, info, r):
        a = np.array(a)
        r = np.array(r)
        info = np.array(info)

        if s.ndim < 2: s = s[:, np.newaxis]
        if a.ndim < 2: a = a[:, np.newaxis]
        if r.ndim < 2: r = r[:, np.newaxis]
        if info.ndim < 2: info = info[:, np.newaxis]
        # a = one_hot(a, self.action_size)
        # print('s : ', s.shape)
        # print('r : ', r.shape)
        self.sess.run(self.model.update_oldpi_op)
        adv = self.sess.run(self.model.advantage, {self.model.tfs: s, self.model.tfdc_r: r, self.model.infotfs: info})

        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update critic
        closs = [self.sess.run([self.model.ctrain_op, self.model.closs], {self.model.tfs: s, self.model.tfdc_r: r,
                                                                          self.model.infotfs: info}) for
                 _ in range(self.model.c_update_steps)]
        # [[None, loss_val], [None, loss_val], ...] -> [loss_val, loss_val, ...,]
        closs = np.array(closs)[:, 1]

        # update actor
        # clipping method, find this is better (OpenAI's paper)
        aloss = [self.sess.run([self.model.atrain_op, self.model.aloss],
                               {self.model.tfs: s, self.model.action: a, self.model.tfadv: adv,
                                self.model.infotfs: info}) for i in range(self.model.a_update_steps)]
        # [[None, loss_val], [None, loss_val], ...] -> [loss_val, loss_val, ...,]
        aloss = np.array(aloss)[:, 1]
        return aloss, closs


class BasicAgent(sc2.BotAI):

    def __init__(self,
                 idx,
                 queue,
                 child_conn):
        super(BasicAgent, self).__init__(is_render=is_render)
        self.is_print = is_debug

        # 게임 에피소드
        self.episode = 0

        # 스텝 당 보상
        self.reward = 0

        # Reset Parameters
        self.is_end_game = False

        # 이전 state
        self.pre_state = None

        # 승률
        self.win_rate = deque(maxlen=mem_maxlen)

        # 보상
        self.total_rewards = deque(maxlen=mem_maxlen)

        # 프로세스 관련
        self.idx = idx
        self.queue = queue
        self.child_conn = child_conn

        # controller 몇 스텝당 의사결정인지
        self.controller_k_step = Controller_K_Step

        # 처음 선택될 서브 폴리시 초기화
        self.selected_sub_policy_id = random.randint(0, 3)
        self.selected_sub_policies = [0, 0, 0, 0]

        # 전처리 Class
        self.obsProc = ObservationProcessor()

        # 관리자
        self.m_unit = None
        self.m_worker = None
        self.m_map_pos = None
        self.m_resource = None
        self.script_scout = None

        # 게임 내 정보
        self.steps_per_game = deque(maxlen=n_games_per_one_epoch)
        self.minerals_per_game = deque(maxlen=n_games_per_one_epoch)
        self.vespene_per_game = deque(maxlen=n_games_per_one_epoch)
        self.nexus_per_game = deque(maxlen=n_games_per_one_epoch)
        self.pylon_per_game = deque(maxlen=n_games_per_one_epoch)
        self.gateway_per_game = deque(maxlen=n_games_per_one_epoch)
        self.cyberneticscore_per_game = deque(maxlen=n_games_per_one_epoch)
        self.assimilator_per_game = deque(maxlen=n_games_per_one_epoch)
        self.probe_per_game = deque(maxlen=n_games_per_one_epoch)
        self.zealot_per_game = deque(maxlen=n_games_per_one_epoch)
        self.stalker_per_game = deque(maxlen=n_games_per_one_epoch)
        self.supply_cap_per_game = deque(maxlen=n_games_per_one_epoch)
        self.supply_left_per_game = deque(maxlen=n_games_per_one_epoch)
        self.supply_used_per_game = deque(maxlen=n_games_per_one_epoch)

    def on_start(self):
        # 게임 끝났는지 여부
        self.is_end_game = False
        # Reset Parameters
        self.pre_state = None
        # 스텝
        self.episode_steps = 0

        # 매니저 초기화
        # UnitManager
        self.m_unit = UnitManger(self)
        # WorkerManager
        self.m_worker = WorkerManager(self)
        # MapPosManager
        self.m_map_pos = Map_Position_Manager(self, debug=is_debug)
        # ResourceManager
        self.m_resource = ResourceManager(self)
        # Script Scout
        self.script_scout = Scout(self)
        self._client.game_step = 23

    def get_current_state_minimap(self):
        # 상태 전처리
        preproc_obs = self.obsProc.processObs(self.episode_steps, self.state.feature_layer, is_print=is_debug)
        # Minimap 추출
        cur_state = preproc_obs['minimap']
        return cur_state

    async def on_step(self, iteration):
        if self.is_end_game:
            return
        if iteration == 0:
            # 일꾼 12개 관리자에 추가
            self.m_worker.init_unit()

        # 디버깅용 출력 값
        if self.is_print:
            # if True:
            print('---*{}*---'.format(iteration))
            print('Selected Policy ID : {} !! '.format(SUB_POLICIES_NAME[self.selected_sub_policy_id]))
            # if self.selected_sub_policy_id == SubPoliciesName.SCOUT.value:
            #     print('SCOUT')
            #     print('Enemy Base : ', self.enemy_start_locations[0])
            # elif self.selected_sub_policy_id == SubPoliciesName.ATTACK.value:
            #     print('ATTACK')
            # else:
            #     print('BUILD & TRAIN')
            #     print('My Base : ', self.start_location)
            # print('Units : ', self.units)
            # print('Nexus : ', self.units(UnitTypeId.NEXUS))
            print('# of Minerals : ', self.minerals)
            print('# of Vespene : ', self.vespene)
            print('# of Nexus : ', self.units(UnitTypeId.NEXUS).amount)
            print('# of Pylons : ', self.units(UnitTypeId.PYLON).amount)
            print('# of Gateways : ', self.units(UnitTypeId.GATEWAY).amount)
            print('# of Cyberneticscores : ', self.units(UnitTypeId.CYBERNETICSCORE).amount)
            print('# of Assimulator : ', self.units(UnitTypeId.ASSIMILATOR).amount)
            print('# of Probes : ', self.units(UnitTypeId.PROBE).amount)
            print('# of Zealots : ', self.units(UnitTypeId.ZEALOT).amount)
            print('# of Stalkers : ', self.units(UnitTypeId.STALKER).amount)
            print('# of supply_left : ', self.supply_left)
            # print('# of supply_used : ', self.supply_used)
            print('# of supply_cap : ', self.supply_cap)

        # 보상 초기화
        self.reward = 0

        # TODO : 죽은 유닛 처리 & 죽은 유닛 보상 처리
        reward_units = self.m_unit.get_reward(self.state.dead_units)
        self.m_worker.update_dead_units(self.state.dead_units)
        self.m_unit.update_units()

        self.reward += reward_units

        try:
            # 자원 관리 단계
            await self.m_resource.on_step(iteration)
            # 일꾼 관리 단계
            worker_actions = self.m_worker.on_step(iteration)
            await self.do_actions(worker_actions)
            # 정찰 유닛 파괴 체크
            is_removed = self.m_worker.remove_scouting_worker(self.state.dead_units)
            if is_removed:
                self.script_scout.on_unit_dead()
            # 정찰 관리 단계
            reward_scout = await self.script_scout.on_step(iteration)
            await self._client.send_debug()
            # 정찰 디버그
            self.reward += reward_scout
        except Exception as e:
            if is_debug:
                print('매니저 예외 발생 : ', e, ' | ', sys.exc_info())

        # Reward 저장
        self.total_rewards.append(self.reward)
        # -- 보상 업데이트 끝 -- #

        # 액션 추출
        cur_state = self.get_current_state_minimap()
        # K-step 마다 Controller 로 Sub-Policy 의 ID 설정
        if iteration % self.controller_k_step == 0 and self.pre_state is not None:
            # 부모 프로세스에게 (s1, r, s2, info, d)를 전달
            # Dense Layer에 추가하는 정보들 시작
            # TODO : 신경망에 추가하는 정보 -> 현재 아군 총 수, 미네랄 총 값, 가스 총 값. ( Normalization )
            info_nexus = self.units(UnitTypeId.NEXUS).owned.amount / 1.0
            info_gateway = self.units(UnitTypeId.GATEWAY).owned.amount / 1.0
            info_probe = self.units(UnitTypeId.PROBE).owned.amount / 1.0
            info_zealot = self.units(UnitTypeId.ZEALOT).owned.amount / 1.0
            info_stalker = self.units(UnitTypeId.STALKER).owned.amount / 1.0
            info_units = self.supply_used / 1.0
            info_left = self.supply_left / 1.0
            info_mineral = self.minerals / 1000.0
            info_vespene = self.vespene / 1000.0

            cur_info = np.array([info_nexus, info_gateway, info_probe, info_zealot, info_stalker, info_units, info_left, info_mineral, info_vespene])
            # Dense Layer에 추가하는 정보들 끝
            self.request_action(pre_state=self.pre_state,
                                reward=self.reward,
                                state=cur_state,
                                info=cur_info,
                                done=False)
            # 모델에서 액션 추출 ( 부모 프로세스에 있는 controller 에서 받음 )
            cur_ctr_actions = self.child_conn.recv()
            # 가능한 액션들 정의
            available_actions = np.ones(shape=[NUM_OF_POLICIES])
            attack_prob = cur_ctr_actions[int(SubPoliciesName.ATTACK.value)]
            # 질럿과 스토커가 없는 경우 가능한 액션에서 제외
            if not self.units(UnitTypeId.ZEALOT).exists and not self.units(UnitTypeId.STALKER).exists:
                available_actions[int(SubPoliciesName.ATTACK.value)] = 0
                # 가능한 액션 처리
                cur_ctr_actions = np.multiply(cur_ctr_actions, available_actions)
                sum_prob = np.sum(cur_ctr_actions)
                for i in range(NUM_OF_POLICIES):
                    cur_ctr_actions[i] += cur_ctr_actions[i] / sum_prob * attack_prob
                sum_prob = np.sum(cur_ctr_actions)
                cur_ctr_actions[np.argmax(cur_ctr_actions)] += 1 - sum_prob
            # print('cur_ctr_actions : ', cur_ctr_actions, ' | ', np.sum(cur_ctr_actions))
            # # 가장 큰 값을 다음 policy 로 지정
            # self.selected_sub_policy_id = np.argmax(cur_ctr_actions)
            # 확률적 액션 선택
            self.selected_sub_policy_id = np.random.choice(np.arange(action_size), p=cur_ctr_actions)
            # print('select : [', self.selected_sub_policy_id)
            self.selected_sub_policies[int(self.selected_sub_policy_id)] += 1
            # 상태 업데이트
            self.pre_state = cur_state

        try:
            if is_camera_move:
                # Camera 이동
                # 정찰 Camera 관리
                if self.selected_sub_policy_id == SubPoliciesName.SCOUT.value:
                    # 적기지 위치 반환
                    await self._client.move_camera(self.enemy_start_locations[0])
                # 공격 Camera 관리
                elif self.selected_sub_policy_id == SubPoliciesName.ATTACK.value:
                    # 내 전투 유닛의 중간 위치 반환
                    if self.units(UnitTypeId.ZEALOT).exists or self.units(UnitTypeId.STALKER).exists:
                        positions = []
                        for unit in self.units(UnitTypeId.ZEALOT):
                            positions.append(unit.position)
                        for unit in self.units(UnitTypeId.STALKER):
                            positions.append(unit.position)
                        center_pos = np.mean(positions, axis=0)
                        if self.is_print: print("Center pos : ", center_pos)
                        next_camera_pos = Point2((center_pos[0], center_pos[1]))

                        await self._client.move_camera(next_camera_pos)
                # 건설 및 생산 Camera 관리
                else:
                    # 내 기지 위치
                    await self._client.move_camera(self.start_location)
        except Exception as e:
            if is_debug:
                print('카메라 이동 예외 발생 : ', e, ' | ', sys.exc_info())

        try:

            '''
                시간증폭 Policy 우선순위
                1. 생산중인 Gateway가 있으면 거기에 시간증폭을 검
                2. 생산중인 Nexus가 있으면 거기에 시간증폭을 검

            '''
            u_nexus = self.units(UnitTypeId.NEXUS).ready.random
            abilities = await self.get_available_abilities(u_nexus)
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                # If has ChronoBoost
                candidate_gateway = []

                if self.units(UnitTypeId.GATEWAY).ready.exists:
                    for u in self.units(UnitTypeId.GATEWAY).ready:
                        if (not u.has_buff(BuffId.CHRONOBOOSTENERGYCOST)) and (not u.noqueue):
                            candidate_gateway.append(u)

                if len(candidate_gateway) > 0:
                    # If has Training Gateway
                    target_gateway = random.choice(candidate_gateway)
                    await self.do(u_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target_gateway))
                else:
                    # If has only Nexus
                    if not u_nexus.noqueue:
                        await self.do(u_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, u_nexus))
            '''
                시간증폭 Policy 끝
            '''

            # 마이크로를 위한 정책들
            if self.selected_sub_policy_id == SubPoliciesName.SCOUT.value:
                if self.units(UnitTypeId.PROBE).exists:
                    target_worker = None
                    target_worker_tag = self.m_worker.get_scouting_worker_tag()
                    # 정찰 일꾼 정하기
                    if target_worker_tag == -1:
                        gas_worker = self.units(UnitTypeId.PROBE).closest_to(self.enemy_start_locations[0])
                        for probe in self.units(UnitTypeId.PROBE):
                            if probe.tag != gas_worker.tag:
                                target_worker = probe
                                self.m_worker.add_scouting_worker(target_worker)
                                break
            elif self.selected_sub_policy_id == SubPoliciesName.ATTACK.value:
                pos_attack = self.enemy_start_locations[0].towards_with_random_angle(
                    Point2(self.m_map_pos.enemy_tile_pos),
                    random.randrange(
                        0,
                        int(
                            self.m_map_pos.eight_tile_half_height)))
                n_attack_units = 0
                attack_actions = []
                for unit in self.units(UnitTypeId.ZEALOT):
                    attack_actions.append(unit.attack(pos_attack))
                    n_attack_units += 1
                for unit in self.units(UnitTypeId.STALKER):
                    attack_actions.append(unit.attack(pos_attack))
                    n_attack_units += 1
                if n_attack_units >= 10:
                    rtn = await self.do_actions(attack_actions)
                    # print('적 기지 공격 Message {} '.format(rtn))
            elif self.selected_sub_policy_id == SubPoliciesName.BUILD.value:
                if not self.units(UnitTypeId.NEXUS).exists:
                    nexus_pos = self.start_location.position
                    if self.can_afford(UnitTypeId.NEXUS):
                        rtn = await self.build(UnitTypeId.NEXUS, near=nexus_pos,
                                               unit=self.m_worker.select_build_worker(nexus_pos))
                else:
                    if not self.units(UnitTypeId.GATEWAY).exists:
                        # 게이트웨이 없을 경우 극 초반 빌드로 진행
                        if not self.units(UnitTypeId.PYLON).exists:
                            # 파일런이 지어지지 않은 경우
                            nexus = self.start_location
                            if self.can_afford(UnitTypeId.PYLON):
                                if nexus.position == Point2((59.5, 27.5)):
                                    pos = Point2((50.98, 26.67))
                                else:
                                    pos = Point2((36.21, 59.16))
                                rtn = await self.build(UnitTypeId.PYLON, near=pos,
                                                       unit=self.m_worker.select_build_worker(pos))
                                if rtn == ActionResult.CantFindPlacementLocation:
                                    if nexus.position == Point2((59.5, 27.5)):
                                        pos = Point2((26.18, 53.73))
                                    else:
                                        pos = Point2((58.498, 33.71))
                                    rtn = await self.build(UnitTypeId.PYLON, near=pos,
                                                           unit=self.m_worker.select_build_worker(pos))
                                    # print('파일런 건설 Message {}'.format(rtn))
                        else:
                            # 파일런이 지어진 경우
                            # 현재 게이트웨이 1순위, 가스 2순위
                            if self.units(UnitTypeId.PYLON).ready.exists:
                                pylon = self.units(UnitTypeId.PYLON).ready.random
                                if self.can_afford(UnitTypeId.GATEWAY):
                                    rtn = await self.build(UnitTypeId.GATEWAY, near=pylon,
                                                           unit=self.m_worker.select_build_worker(pylon.position))
                                    # print('게이트웨이 건설 Message {} '.format(rtn))
                    else:
                        # 게이트웨이가 지어진 상태인 경우
                        if not self.units(UnitTypeId.CYBERNETICSCORE).exists and not self.already_pending(
                                UnitTypeId.CYBERNETICSCORE):
                            if self.units(UnitTypeId.GATEWAY).ready.exists and self.units(
                                    UnitTypeId.ASSIMILATOR).exists:  # 중요! 가스가 지어지지 않았으면 가스가 지어질 때 까지 대기함
                                pylon = self.units(UnitTypeId.PYLON).ready.random
                                if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                                    rtn = await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon,
                                                           unit=self.m_worker.select_build_worker(pylon.position))
                                    # print('사이버네틱스코어 건설 Message {} '.format(rtn))

                        if self.supply_left <= int(self.units(UnitTypeId.GATEWAY).amount * 2 + 3):
                            # 추가적으로 파일런이 필요한 경우
                            # nexus = self.units(UnitTypeId.NEXUS).ready.random
                            nexus = self.start_location
                            if self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON):
                                if nexus.position == Point2((59.5, 27.5)):
                                    pos = Point2((50.98, 26.6))
                                else:
                                    pos = Point2((36.21, 59.16))
                                rtn = await self.build(UnitTypeId.PYLON, near=pos,
                                                       unit=self.m_worker.select_build_worker(pos))
                                if rtn == ActionResult.CantFindPlacementLocation:
                                    if nexus.position == Point2((59.5, 27.5)):
                                        pos = Point2((26.18, 53.73))
                                    else:
                                        pos = Point2((58.498, 33.71))
                                    rtn = await self.build(UnitTypeId.PYLON, near=pos,
                                                           unit=self.m_worker.select_build_worker(pos))
                                # print('파일런 건설 Message {}'.format(rtn))

                        if self.minerals >= 300 and not self.already_pending(UnitTypeId.GATEWAY) and self.can_afford(
                                UnitTypeId.GATEWAY):
                            pylon = self.units(UnitTypeId.PYLON).ready.random
                            rtn = await self.build(UnitTypeId.GATEWAY, near=pylon,
                                                   unit=self.m_worker.select_build_worker(pylon.position))

                            # print('게이트웨이 건설 Message {} '.format(rtn))

                    if self.units(UnitTypeId.GATEWAY).exists and not self.units(UnitTypeId.ASSIMILATOR).exists:
                        if not self.already_pending(UnitTypeId.ASSIMILATOR) and self.can_afford(UnitTypeId.ASSIMILATOR):
                            nexus = self.units(UnitTypeId.NEXUS).ready.random
                            target = self.state.vespene_geyser.closest_to(nexus)
                            probe = self.m_worker.select_build_worker(target.position)
                            rtn = await self.do(probe.build(UnitTypeId.ASSIMILATOR, target))
                            # print('가스 건설 Message {} '.format(rtn))

                    if self.units(UnitTypeId.GATEWAY).amount > 2 and self.units(
                            UnitTypeId.ASSIMILATOR).ready.amount == 1:
                        if not self.already_pending(UnitTypeId.ASSIMILATOR) and self.can_afford(UnitTypeId.ASSIMILATOR):
                            nexus = self.units(UnitTypeId.NEXUS).ready.random
                            target = self.state.vespene_geyser.closest_to(nexus)
                            probe = self.m_worker.select_build_worker(target.position)
                            rtn = await self.do(probe.build(UnitTypeId.ASSIMILATOR, target))
                            # print('가스 건설 Message {} '.format(rtn))
            elif self.selected_sub_policy_id == SubPoliciesName.TRAIN.value:
                if self.supply_left >= 2:
                    if self.units(UnitTypeId.GATEWAY).exists and self.units(
                            UnitTypeId.CYBERNETICSCORE).exists and self.units(UnitTypeId.CYBERNETICSCORE).ready:
                        gateway = self.units(UnitTypeId.GATEWAY).ready.random
                        if self.can_afford(UnitTypeId.STALKER):
                            # print('스토커 생성')
                            rtn = await self.do(gateway.train(UnitTypeId.STALKER))
                if self.supply_left >= 2:
                    if self.units(UnitTypeId.PYLON).exists and self.units(UnitTypeId.GATEWAY).exists and self.units(
                            UnitTypeId.GATEWAY).ready:
                        gateway = self.units(UnitTypeId.GATEWAY).ready.random
                        if self.can_afford(UnitTypeId.ZEALOT):
                            # print('질럿 생성')
                            rtn = await self.do(gateway.train(UnitTypeId.ZEALOT))
                            # print('질럿 생성 Message {} '.format(rtn))
                if self.supply_left >= 1 and self.units(UnitTypeId.PROBE).amount <= 30:
                    if self.units(UnitTypeId.NEXUS).exists:
                        nexus = None
                        for n in self.units(UnitTypeId.NEXUS):
                            if n.is_mine and n.is_ready:
                                nexus = n
                                break
                        # nexus = self.units(UnitTypeId.NEXUS).ready.random
                        if self.can_afford(UnitTypeId.PROBE) and nexus is not None:
                            # print('프로브 생성')
                            rtn = await self.do(nexus.train(UnitTypeId.PROBE))
                            # print('프로브 생성 Message {} '.format(rtn))
        except Exception as e:
            if is_debug:
                print('스크립트 파트 예외 발생 : ', e, ' | ', sys.exc_info())

        # 이전 상태 초기화
        if self.pre_state is None:
            # TODO : on_start에서 state 못받아와서.. (self.state.feature_layer 접근 불가.)
            self.pre_state = cur_state
            self.episode_steps += 1
            return

        # 에피소드 스텝 수 증가
        self.episode_steps += 1

    def request_action(self, pre_state, reward, state, info, done):
        self.queue.put([self.idx, "OnStep", [pre_state, reward, state, info, done]])

    def send_result(self, total_rewards, win_rate, selected_policies, total_game_info):
        self.queue.put([self.idx, "Result", [total_rewards, win_rate, selected_policies, total_game_info]])

    def on_end(self, result):
        # 터미널 상태 체크
        self.is_end_game = True
        str_win = None
        is_win = 0

        self.reward = 0

        # 게임 결과를 통해 보상 정의
        if not result == Result.Tie and not result == Result.Defeat:
            # 승리시 보상 1
            str_win = "Win"
            is_win = 1
            self.reward = 1
            # 리플레이 게임 저장
            if is_save_win_game_replay:
                dt = datetime.datetime.now()
                # TODO : async 함수 호출 시 처리
                self._client.save_replay(save_replay_path + dt.strftime('%m%d%H%M%S'))
        else:
            # 패배 및 드로우 시 보상 -1
            str_win = "Loss or Draw"
            is_win = 0
            self.reward = -1

        # 공유 메모리에 승/패 유무 저장
        self.win_rate.append(is_win)
        # 보상 저장
        self.total_rewards.append(self.reward)

        if self.pre_state is not None:
            # 현재 상태와 다음 액션 선택
            cur_state = self.get_current_state_minimap()

            # Dense Layer에 추가하는 정보들 시작
            # TODO : 신경망에 추가하는 정보 -> 현재 아군 유닛의 총 수, 미네랄 총 값, 가스 총 값. ( Normalization )
            info_nexus = self.units(UnitTypeId.NEXUS).owned.amount / 1.0
            info_gateway = self.units(UnitTypeId.GATEWAY).owned.amount / 1.0
            info_probe = self.units(UnitTypeId.PROBE).owned.amount / 1.0
            info_zealot = self.units(UnitTypeId.ZEALOT).owned.amount / 1.0
            info_stalker = self.units(UnitTypeId.STALKER).owned.amount / 1.0
            info_units = self.supply_used / 1.0
            info_left = self.supply_left / 1.0
            info_mineral = self.minerals / 1000.0
            info_vespene = self.vespene / 1000.0

            cur_info = np.array([info_nexus, info_gateway, info_probe, info_zealot, info_stalker, info_units, info_left, info_mineral, info_vespene])
            # Dense Layer에 추가하는 정보들 끝
            self.request_action(pre_state=self.pre_state,
                                reward=self.reward,
                                state=cur_state,
                                info=cur_info,
                                done=True)
            cur_ctr_actions = self.child_conn.recv()
            # 확률적 액션 선택
            self.selected_sub_policy_id = np.random.choice(np.arange(action_size), p=cur_ctr_actions)
            self.selected_sub_policies[int(self.selected_sub_policy_id)] += 1

        # 에피소드 증가
        self.episode += 1

        if self.is_print:
            print('Episode : {}    Is Win : {}'.format(self.episode, str_win))

        self.steps_per_game.append(self.episode_steps)
        self.minerals_per_game.append(self.minerals)
        self.vespene_per_game.append(self.vespene)
        self.nexus_per_game.append(self.units(UnitTypeId.NEXUS).amount)
        self.pylon_per_game.append(self.units(UnitTypeId.PYLON).amount)
        self.gateway_per_game.append(self.units(UnitTypeId.GATEWAY).amount)
        self.cyberneticscore_per_game.append(self.units(UnitTypeId.CYBERNETICSCORE).amount)
        self.assimilator_per_game.append(self.units(UnitTypeId.ASSIMILATOR).amount)
        self.probe_per_game.append(self.units(UnitTypeId.PROBE).amount)
        self.zealot_per_game.append(self.units(UnitTypeId.ZEALOT).amount)
        self.stalker_per_game.append(self.units(UnitTypeId.STALKER).amount)
        self.supply_cap_per_game.append(self.supply_cap)
        self.supply_left_per_game.append(self.supply_left)
        self.supply_used_per_game.append(self.supply_used)

        # Terminal 샘플 저장
        if self.episode == n_games_per_one_epoch:
            total_game_info = {
                'steps': np.mean(self.steps_per_game),
                'minerals': np.mean(self.minerals_per_game),
                'vespene': np.mean(self.vespene_per_game),
                'nexus': np.mean(self.nexus_per_game),
                'pylon': np.mean(self.pylon_per_game),
                'gateway': np.mean(self.gateway_per_game),
                'cyberneticscore': np.mean(self.cyberneticscore_per_game),
                'assimilator': np.mean(self.assimilator_per_game),
                'probe': np.mean(self.probe_per_game),
                'zealot': np.mean(self.zealot_per_game),
                'stalker': np.mean(self.stalker_per_game),
                'supply_cap': np.mean(self.supply_cap_per_game),
                'supply_left': np.mean(self.supply_left_per_game),
                'supply_used': np.mean(self.supply_used_per_game)
            }

            self.send_result(np.sum(self.total_rewards), np.mean(self.win_rate), self.selected_sub_policies,
                             total_game_info)


class SC2Env(Process):
    def __init__(self, **kwargs):
        super(SC2Env, self).__init__()
        self.idx = kwargs['idx']
        self.queue = kwargs['queue']
        self.child_conn = kwargs['child_conn']
        self.difficulty = kwargs['difficulty']

    def run(self):
        super(SC2Env, self).run()

        sc2.run_game(sc2.maps.get(map_name), [
            Bot(my_race, BasicAgent(idx=self.idx,
                                    queue=self.queue,
                                    child_conn=self.child_conn)),
            Computer(enemy_race, self.difficulty)
        ],
                     feature_setting=FeatureSetting(
                         screen=(screen_size_px, screen_size_px),
                         minimap=(minimap_size_px, minimap_size_px)
                     ),
                     realtime=realtime, game_time_limit=game_time_limit,
                     )


if __name__ == '__main__':
    tf.reset_default_graph()

    # 시작 시간
    start_time = time.time()

    # 세션
    # GPU 사용량 나누기
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()
    # GPU 램 얼마나 먹는지 확인하기 위한 용도
    # config.gpu_options.allow_growth = True

    if not is_debug:
        logging.getLogger().setLevel(logging.ERROR)

    with tf.Session(config=config) as sess:
        # Controller 생성
        controller = Controller(model_name="Single",
                                sess=sess,
                                policy=PPO)

        # TODO : Sub-Policy 생성
        # sub_policies = SubPolicy()

        # 세션 초기화
        sess.run(tf.global_variables_initializer())

        # 적군 Level 올리기 -> index
        difficulty = 0

        # 레벨 별 승률
        epoch_win_rates = deque(maxlen=20)

        for epoch in range(n_game_epochs):
            save_path_dir = save_path + str(difficulty) + '/'

            n_success_workers = 0
            queue = Queue()
            workers = deque(maxlen=num_worker)
            parent_conns = deque(maxlen=num_worker)
            win_rates = deque(maxlen=num_worker)
            total_rewards = deque(maxlen=num_worker)
            total_selected_policies = np.array([0, 0, 0, 0])
            last_session = dict()  # 타임아웃을 체크하기 위한 변수
            end_client = 0
            total_game_info = {
                'steps': 0.,
                'minerals': 0.,
                'vespene': 0.,
                'nexus': 0.,
                'pylon': 0.,
                'gateway': 0.,
                'cyberneticscore': 0.,
                'assimilator': 0.,
                'probe': 0.,
                'zealot': 0.,
                'stalker': 0.,
                'supply_cap': 0.,
                'supply_left': 0.,
                'supply_used': 0.
            }

            for idx in range(num_worker):
                parent_conn, child_conn = Pipe()

                # env_args를 정의한다.
                env_args = dict(
                    idx=idx,
                    queue=queue,
                    child_conn=child_conn,
                    difficulty=enemy_difficulties[difficulty]
                )
                worker = SC2Env(**env_args)
                worker.start()
                workers.append(worker)
                last_session[idx] = int(time.time())
                parent_conns.append(parent_conn)

            while True:
                is_time_out = False
                # 자식 프로세서에서 들어오는 값들을 대기
                while queue.empty() and not is_time_out:  # Wait for worker's state
                    # print('queue', time.time())
                    # 타임아웃 된 자식 프로세스 종료
                    timestamp_now = int(time.time())
                    env_sessions_idx = len(last_session.keys())
                    for idx in range(env_sessions_idx):
                        # 30초이상이 되면 강제종료
                        if (last_session[idx] is not None) and (timestamp_now - last_session[idx]) >= 30:
                            print('force terminate worker {}'.format(idx))
                            workers[idx].terminate()
                            workers[idx] = None
                            last_session[idx] = None
                            end_client += 1
                            is_time_out = True
                            break
                    continue

                if not is_time_out:
                    # Received some data
                    idx, command, parameter = queue.get()

                    if idx in last_session.keys():
                        last_session[idx] = int(time.time())

                    # 액션 요청 처리 : (s1, r, s2, d)를 받고 action 전달
                    if command == "OnStep":
                        pre_state, reward, state, info, done = parameter

                        action = controller.choose_action(state, info)
                        assert pre_state is not None, print('pre_state : ', pre_state)
                        assert action is not None, print('action : ', action)
                        assert reward is not None, print('reward : ', reward)
                        assert state is not None, print('state : ', state)
                        assert info is not None, print('info : ', info)
                        assert done is not None, print('done : ', done)

                        controller.append_sample(pre_state, action, reward, state, info, done)

                        try:
                            parent_conns[idx].send(action)
                        except:  # 강제종료 된 프로세스에는 송신시 에러남
                            if is_debug:
                                print('강제종료 된 프로세스에는 송신시 에러남')

                    # 게임 종료 요청 처리
                    elif command == "Result":
                        total_reward, win_rate, selected_policies, game_info = parameter
                        # Eposide 당 보상을 나눠 줌.
                        total_rewards.append(total_reward)
                        win_rates.append(win_rate)
                        total_selected_policies += selected_policies
                        for k, v in total_game_info.items():
                            total_game_info[k] += game_info[k]

                        print('Idx : ', idx, ' | total_rewards : ', np.mean(total_rewards), ' | win_rates : ',
                              np.mean(win_rates), ' | selected_policies : ', selected_policies)

                        end_client += 1
                        n_success_workers += 1

                # 모든 자식 프로세서가 끝난 경우 처리
                if end_client == num_worker:
                    alosses = []
                    closses = []

                    # 학습 시작
                    if epoch % print_interval == 0 and batch_size < len(controller.memory) and is_train:
                        for _ in range(n_learning):
                            # 학습
                            aloss, closs = controller.train_model()
                            if aloss is not None: alosses.append(aloss)
                            closses.append(closs)

                    # 학습 결과 프린트
                    print(
                        'Episodes({}) | success worker({}/{}) : rewards: {:.6f}    aloss: {:.4f}     closs: {:.4f}     win_rate: {:.4f}     memory: {}     total_selected_policies: {}'.format(
                            epoch, n_success_workers, num_worker,
                            np.mean(total_rewards),
                            np.mean(alosses),
                            np.mean(closses),
                            np.mean(win_rates),
                            len(controller.memory),
                            total_selected_policies / (n_games_per_one_epoch * n_success_workers)))

                    # 모든 게임 정보 종합 처리
                    for k, game_info in total_game_info.items():
                        total_game_info[k] /= (n_games_per_one_epoch * n_success_workers)

                    # Summary 에 저장
                    controller.write_summray(reward=np.mean(total_rewards),
                                             aloss=np.mean(alosses),
                                             closs=np.mean(closses),
                                             win_rate=np.mean(win_rates),
                                             game_level=enemy_difficulties[difficulty].value,
                                             actions=total_selected_policies / (n_games_per_one_epoch * n_success_workers),
                                             memory_size=len(controller.memory),
                                             game_info=total_game_info,
                                             episode=epoch)

                    # 모델 저장
                    if epoch % save_interval == 0 and is_train:
                        controller.save_model(save_path_dir)
                        print("Save Model. EPISODE : {}".format(epoch))

                    # epoch 당 승률 저장
                    epoch_win_rates.append(np.mean(win_rates))
                    print("#{} epoch mean win rates {}".format(epoch, np.mean(epoch_win_rates)))

                    # 승률이 93% 이상일 경우 다음 레벨로 올라가기
                    if len(epoch_win_rates) == 20 and np.mean(epoch_win_rates) >= 0.93:
                        # print("Level Up ------------------")
                        epoch_win_rates.clear()
                        controller.save_model(save_path_dir)
                        # 7레벨 까지 레벨 올리기
                        if difficulty < 6:
                            difficulty += 1
                            game_time_limit += 300

                    # 자원 처리
                    for worker in workers:
                        if worker is not None:
                            worker.terminate()
                    os.system('pkill -9 -ef SC2')
                    os.system('pkill -9 -ef SC2')

                    workers.clear()
                    parent_conns.clear()

                    del last_session
                    del alosses
                    del closses
                    del workers
                    del parent_conns
                    del queue
                    del total_rewards
                    del win_rates
                    del total_game_info
                    del save_path_dir

                    break
