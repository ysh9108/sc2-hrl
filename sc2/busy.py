from sc2.constants import *
from sc2.position import Point2
import numpy, cv2

BUILDINGS_HAS_QUEUE = [
    NEXUS,
    GATEWAY,
    CYBERNETICSCORE,
    STARGATE,
    ROBOTICSFACILITY,
    ROBOTICSBAY,
    FLEETBEACON,
    TEMPLARARCHIVE,
    DARKSHRINE
]

class BusyFeature:

    def __init__(self, state, map_size, target_size, type=None):

        assert type in ['screen', 'minimap']

        self.state = state

        self.size = target_size
        self.map_size = map_size
        self.np_data = None
        self.rgb_data = None

        self._ratio_y = self.size[1] / self.map_size.y
        self._ratio_x = self.size[0] / self.map_size.x

        w_half = self.map_size.x // 2
        h_half = self.map_size.y // 2
        w_gap = h_gap = 0
        if w_half < h_half:
            h_gap = h_half - w_half
        elif w_half > h_half:
            w_gap = w_gap - h_gap

        if type == 'minimap':
            raw = numpy.zeros((self.size[1], self.size[0]), dtype=numpy.int32)

            units = self.state.units
            for u in units.structure.owned:
                if u.type_id in BUILDINGS_HAS_QUEUE:
                    half_size = int(u.radius / 1.2 * self._ratio_x)
                    raw = cv2.rectangle(raw,
                                        (int((u.position.x - half_size) * self._ratio_x), int((self.map_size.y - u.position.y - half_size) * self._ratio_y)),
                                        (int((u.position.x + half_size) * self._ratio_x), int((self.map_size.y - u.position.y + half_size) * self._ratio_y)),
                                        len(u.orders), -1)
            if w_gap != 0 or h_gap != 0:
                if self.size[0] > self.size[1]:
                    # 가로가 더 긴 경우
                    w_half = self.size[0] // 2
                    h_half = self.size[1] // 2
                    gap = w_half - h_half
                    raw = cv2.copyMakeBorder(raw, gap, gap, 0, 0, cv2.BORDER_CONSTANT, value=0)
                else:
                    # 세로가 더 긴 경우
                    w_half = self.size[0] // 2
                    h_half = self.size[1] // 2
                    gap = h_half - w_half
                    raw = cv2.copyMakeBorder(raw, 0, 0, 0, gap * 2, cv2.BORDER_CONSTANT, value=0)
            self.np_data = raw

        else:
            raw = numpy.zeros((self.size[1], self.size[0]), dtype=numpy.uint8)
            units = self.state.units

            camera = self.state.observation.raw_data.player.camera
            width_camera = 24
            camera =Point2((camera.x - width_camera // 2, camera.y - width_camera // 2))


            for u in units.structure.owned:
                if u.type_id in BUILDINGS_HAS_QUEUE:
                    if not u.position.distance_to(Point2((camera.x, camera.y))) <= width_camera * 3:
                        continue

                    relative_postion = ((u.position - camera) * (84/24))
                    raw = cv2.circle(raw,
                                        (int(relative_postion.x), self.size[1] - int(relative_postion.y)),
                                        radius=int(u.radius * 3.6), color=len(u.orders), thickness=-1)
            self.np_data = raw

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def numpy(self):
        return self.np_data

    @property
    def rgb(self):
        if self.rgb_data is None:
            _np_data = (self.np_data / 5 * 255).astype(numpy.uint8)
            self.rgb_data = cv2.applyColorMap(_np_data, cv2.COLORMAP_HOT)
            self.rgb_data = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB)
        return self.rgb_data
