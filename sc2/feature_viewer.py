import threading
from queue import Queue
import numpy, cv2

import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    from pygame.locals import *

class FeatureViewer(threading.Thread):
    def __init__(self, obs_queue):
        super(FeatureViewer, self).__init__()
        self.obs_queue = obs_queue

        black_screen = numpy.zeros([100, 100, 3], dtype=numpy.uint8)
        black_screen = pygame.image.frombuffer(black_screen, black_screen.shape[1::-1], 'RGB')

        self.surfaces = dict()
        self.surfaces['screen'] = dict()
        self.surfaces['minimap'] = dict()
        screen_features = ['height_map', 'visibility_map', 'creep', 'power', 'player_id', 'unit_type', 'selected',
                           'unit_hit_points', 'unit_hit_points_ratio', 'unit_energy', 'unit_energy_ratio',
                           'unit_shields', 'unit_shields_ratio', 'player_relative', 'unit_density_aa', 'unit_density',
                           'structure_busy'
                            ]
        minimap_features = ['height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative', 'selected', 'structure_busy', 'unit_type']
        for layer in screen_features:
            self.surfaces['screen'][layer] = black_screen
        for layer in minimap_features:
            self.surfaces['minimap'][layer] = black_screen

        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 11, bold=True)

    def run(self):

        pygame.init()
        pygame.display.set_caption('FeatureLayer Display')
        screen = pygame.display.set_mode((650, 580))
        clock = pygame.time.Clock()

        run = True
        drawed_once = False

        screen.fill((15, 15, 15))
        while run:
            for event in pygame.event.get():
                if event.type == QUIT:
                    run = False  # Exits the loop. Not sure if 'exit()' was defined
                    break

            if not self.obs_queue.empty():
                obs = self.obs_queue.get()

                for layer in list(self.surfaces['screen'].keys()):
                    rgb_image = cv2.resize(obs.feature_layer['screen'][layer].rgb, (100, 100))
                    self.surfaces['screen'][layer] = pygame.image.frombuffer(rgb_image, rgb_image.shape[1::-1], 'RGB')
                for layer in list(self.surfaces['minimap'].keys()):
                    rgb_image = cv2.resize(obs.feature_layer['minimap'][layer].rgb, (100, 100))
                    self.surfaces['minimap'][layer] = pygame.image.frombuffer(rgb_image, rgb_image.shape[1::-1], 'RGB')

                # print(obs)
                self.obs_queue.task_done()


            for i, layer in enumerate(self.surfaces['minimap']):
                screen.blit(self.surfaces['minimap'][layer], ((100 * (i % 6)) + (i % 6 * 10), 100 * (i // 6) + ((i // 6) * 15)))
                if not drawed_once:
                    txt_surf = self.font.render(layer, False, (255, 255, 255))
                    screen.blit(txt_surf, ((100 * (i % 6)) + (i % 6 * 10), 100 * (i // 6) + ((i // 6) * 15) + 100))

            for i, layer in enumerate(self.surfaces['screen']):
                screen.blit(self.surfaces['screen'][layer], ((100 * (i % 6)) + (i % 6 * 10), 100 * (i // 6) + ((i // 6) * 15)  + 230))
                if not drawed_once:
                    txt_surf = self.font.render(layer, False, (255, 255, 255))
                    screen.blit(txt_surf, ((100 * (i % 6)) + (i % 6 * 10), 100 * (i // 6) + ((i // 6) * 15) + 100  + 230))

            drawed_once = True

            pygame.display.update()
            clock.tick(30)
        pygame.quit()