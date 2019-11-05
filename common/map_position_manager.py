import sc2
from sc2.constants import *
from sc2.position import Point2

class Map_Position_Manager:
    def __init__(self, agent, debug=False):
        self.agent = agent
        self.debug = debug

        self.map_width = self.agent.game_info.playable_area[2] - self.agent.game_info.playable_area[0]
        self.map_height = self.agent.game_info.playable_area[3] - self.agent.game_info.playable_area[1]

        self.eight_tile_half_width = self.agent.game_info.playable_area[0] + self.map_width / 3 / 2
        self.eight_tile_half_height= self.agent.game_info.playable_area[1] + self.map_height/ 3 / 2

        # Index of Eight-Tile-Position
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.eight_tile_pos = [
            (self.eight_tile_half_width, self.eight_tile_half_height), # 0번째 Pos
            (self.eight_tile_half_width + self.map_width / 3, self.eight_tile_half_height),  # 1번째 Pos
            (self.eight_tile_half_width + self.map_width / 3 * 2, self.eight_tile_half_height),  # 2번째 Pos
            (self.eight_tile_half_width, self.eight_tile_half_height + self.map_height / 3),  # 3번째 Pos
            (self.eight_tile_half_width + self.map_width / 3, self.eight_tile_half_height + self.map_height / 3),  # 4번째 Pos
            (self.eight_tile_half_width + self.map_width / 3 * 2, self.eight_tile_half_height + self.map_height / 3),  # 5번째 Pos
            (self.eight_tile_half_width, self.eight_tile_half_height + self.map_height / 3 * 2),  # 6번째 Pos
            (self.eight_tile_half_width + self.map_width / 3, self.eight_tile_half_height + self.map_height / 3 * 2),  # 7번째 Pos
            (self.eight_tile_half_width + self.map_width / 3 * 2, self.eight_tile_half_height + self.map_height / 3 * 2)  # 8번째 Pos
        ]

        self.enemy_tile_pos = None
        min_dis = 9999
        for idx, pos in enumerate(self.eight_tile_pos):
            cur_dis = Point2(pos).distance_to_point2(self.agent.enemy_start_locations[0])
            if min_dis > cur_dis:
                self.enemy_tile_pos = pos
                min_dis = cur_dis

        if self.debug:
            print('******playable_area : ', self.agent.game_info.playable_area)
            print('my Base : ', self.agent.start_location)
            print('enemy : ', self.agent.enemy_start_locations[0])

            print('Map Size Width : ', self.map_width, ' Height : ', self.map_height)
            print('[Eight-Tile-Pos]')
            for pos in self.eight_tile_pos:
                print('Pos : ', pos)