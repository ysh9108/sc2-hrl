from multiprocessing import Process, Pipe
from pysc2.env import sc2_env, available_actions_printer

# below (worker, CloudpickleWrapper, SubprocVecEnv) copied from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some sc2 specific modifications
def worker(remote, env_fn_wrapper):
    """
    Handling the:
    action -> [action] and  [timestep] -> timestep
    single-player conversions here
    """
    env = env_fn_wrapper.x()
    while True:
        cmd, action = remote.recv()
        if cmd == 'step':
            timesteps = env.on_step([action])
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'reset':
            timesteps = env.reset()
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError



class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv:
    def __init__(self, env_fns):
        # self.remotes : 보통 receiver
        # self.work_remotes : 보통 sender
        # 생성할 환경 개수만큼 Pipe와 Process를 생성한 후 시작시킨다.
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.n_envs = n_envs

    def _step_or_reset(self, command, actions=None):
        actions = actions or [None] * self.n_envs
        for remote, action in zip(self.remotes, actions):
            remote.send((command, action))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def step(self, actions):
        return self._step_or_reset("step", actions)

    def reset(self):
        return self._step_or_reset("reset", None)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def reset_done_envs(self):
        pass
