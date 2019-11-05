from multiprocessing import Pipe, Process

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.client import FeatureSetting


class Agent(sc2.BotAI):

    def __init__(self, is_render=False):
        super(Agent, self).__init__(is_render=is_render)

    def on_start(self):
        pass

    async def on_step(self, iteration):
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
            Bot(Race.Terran, Agent(is_render=self.is_render)),
            Computer(Race.Terran, Difficulty.Easy)
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

