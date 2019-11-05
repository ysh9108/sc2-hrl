
class Reward:

    # 생성 및 파괴 시 리워드
    NEXUS = 0.0009
    PYLON = 0.000003
    GATEWAY = 0.000004
    CYBERNETICSCORE = 0.000005
    ASSIMILATOR = 0.000005

    PROBE = 0.000002
    ZEALOT = 0.00001
    STALKER = 0.00002

    '''
    Reward
    1. 적군 영역 안에 들어오면  +0.00000001
    2. 새로운 건물 발견 시      +0.0000005
    3. 새로운 유닛 발견 시      +0.0000002
    4. 정찰유닛이 죽으면        -0.00004
    '''
    # SCOUT
    SCOUT_IN_ENEMY_AREA = 0.00000001
    SCOUT_ENEMY_NEW_BUILDING = 0.0000005
    SCOUT_ENEMY_NEW_UNITS = 0.0000002
    SCOUT_DEAD = -0.00004

    # ATTACK
    # ENEMY_BUILDING = 0.1
    ATTACK_DIS = 1000000
    ATTACK_PENALTY = -0.0001

    def __init__(self):
        pass