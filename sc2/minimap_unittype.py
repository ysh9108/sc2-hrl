from sc2.constants import *
from .colors import UNITTYPE_PALETTE
from .onehot import UNIT_TYPE_ONEHOT
from sc2.position import Point2
import numpy, cv2

class UnitTypeFeature:

    def __init__(self, state, map_size, target_size, type=None):

        assert type in ['minimap']

        self.state = state

        self.size = target_size
        self.map_size = map_size
        self.np_data = None
        self.rgb_data = None
        self.onehot_data = None

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

            for u in units:
                half_size = int(u.radius / 1.2 * self._ratio_x)

                raw = cv2.rectangle(raw,
                                    (int((u.position.x - half_size) * self._ratio_x), int((self.map_size.y - u.position.y - half_size) * self._ratio_y)),
                                    (int((u.position.x + half_size) * self._ratio_x), int((self.map_size.y - u.position.y + half_size) * self._ratio_y)),
                                    u.type_id.value, -1)

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


    @property
    def one_hot(self):
        if self.onehot_data is not None:
            return self.onehot_data

        self.onehot_data = numpy.take(UNIT_TYPE_ONEHOT, self.np_data, axis=0, mode='clip').astype(numpy.uint8)
        return self.onehot_data


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
        if self.rgb_data is not None:
            return self.rgb_data
        self.rgb_data = numpy.take(UNITTYPE_PALETTE, self.np_data, axis=0, mode='clip').astype(numpy.uint8)
        return self.rgb_data