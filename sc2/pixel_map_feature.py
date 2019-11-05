from typing import Callable, Set, FrozenSet, List
from .position import Point2
from .colors import CREEP_PALETTE, CAMERA_PALETTE, PLAYER_ABSOLUTE_PALETTE, POWER_PALETTE, VISIBILITY_PALETTE, UNITTYPE_PALETTE, EFFECTS_PALETTE, PLAYER_RELATIVE_PALETTE, SELECTED_PALETTE, EFFECTS_PALETTE, winter, hot
from .onehot import UNIT_TYPE_ONEHOT, VISIBILITY_ONEHOT, PLAYER_RELATIVE_ONEHOT, EFFECT_ONEHOT
import numpy

class FeatureScreenDataTypes():
    height_map = numpy.uint8
    visibility_map = numpy.uint8
    creep = numpy.bool
    power = numpy.bool
    player_id = numpy.uint8
    unit_type = numpy.int32
    selected = numpy.bool
    unit_hit_points = numpy.int32
    unit_hit_points_ratio = numpy.uint8
    unit_energy = numpy.uint32
    unit_energy_ratio = numpy.uint8
    unit_shields = numpy.uint32
    unit_shields_ratio = numpy.uint8
    player_relative = numpy.uint8
    unit_density_aa = numpy.uint8
    unit_density = numpy.uint8
    effects = numpy.uint8


class FeatureMinimapDataTypes():
    height_map = numpy.uint8
    visibility_map = numpy.uint8
    creep = numpy.bool
    camera = numpy.bool
    player_id = numpy.uint8
    player_relative = numpy.uint8
    selected = numpy.bool


class FeatureDataTypes():
    screen = FeatureScreenDataTypes
    minimap = FeatureMinimapDataTypes


class PixelMapFeature(object):
    def __init__(self, proto, type, attr):
        self._proto = proto
        self.type = type
        self.attr = attr
        self.data = bytearray(self._proto.data)
        _dtype = getattr(getattr(FeatureDataTypes, type), attr)

        self.np_data = None
        self.rgb_data = None
        self.onehot_data = None
        self._numpized()

    @property
    def width(self):
        return self._proto.size.x

    @property
    def height(self):
        return self._proto.size.y

    @property
    def bits_per_pixel(self):
        return self._proto.bits_per_pixel

    @property
    def raw(self):
        return self.data

    @property
    def numpy(self):
        if self.np_data is None:
            self._numpized()
        return self.np_data

    @property
    def one_hot(self):
        assert self.attr in ['unit_type', 'visibility_map', 'player_relative', 'effects']

        if self.np_data is None:
            self._numpized()

        if self.attr == 'unit_type':
            self.onehot_data = numpy.take(UNIT_TYPE_ONEHOT, self.np_data, axis=0, mode='clip').astype(numpy.uint8)
        elif self.attr == 'visibility_map':
            self.onehot_data = numpy.take(VISIBILITY_ONEHOT, self.np_data, axis=0, mode='clip').astype(numpy.uint8)
        elif self.attr == 'player_relative':
            self.onehot_data = numpy.take(PLAYER_RELATIVE_ONEHOT, self.np_data, axis=0, mode='clip').astype(numpy.uint8)
        elif self.attr == 'effects':
            self.onehot_data = numpy.take(EFFECT_ONEHOT, self.np_data, axis=0, mode='clip').astype(numpy.uint8)

        return self.onehot_data


    @property
    def rgb(self):
        if self.rgb_data is None:
            self._colorize()
        return self.rgb_data

    def _numpized(self):
        _dtype = getattr(getattr(FeatureDataTypes, self.type), self.attr)
        if _dtype != numpy.bool:
            data_np = numpy.frombuffer(self.data, dtype=_dtype)
            data_np = data_np.reshape([self._proto.size.y, self._proto.size.x])
            self.np_data = data_np
        else:
            data_np = numpy.unpackbits(self.data)
            data_np = data_np.reshape([self._proto.size.y, self._proto.size.x])
            self.np_data = data_np

    def _colorize(self):
        # feature의 특성에 맞게 색칠
        # https://github.com/cilab-matser/CILAB-sc2/blob/master/docs/feature.md

        if self.rgb_data is not None:
            return self.rgb_data

        if self.attr == 'height_map':
            palette = winter(256)
        elif self.attr == 'visibility_map':
            palette = VISIBILITY_PALETTE
        elif self.attr == 'creep':
            palette = CREEP_PALETTE
        elif self.attr == 'power':
            palette = POWER_PALETTE
        elif self.attr == 'unit_type':
            palette = UNITTYPE_PALETTE
        elif self.attr == 'unit_hit_points':
            palette = hot(1600)
        elif self.attr == 'unit_hit_points_ratio':
            palette = hot(256)
        elif self.attr == 'unit_energy':
            palette = hot(1000)
        elif self.attr == 'unit_energy_ratio':
            palette = hot(256)
        elif self.attr == 'unit_shields':
            palette = hot(1000)
        elif self.attr == 'unit_shields_ratio':
            palette = hot(256)
        elif self.attr == 'unit_density':
            palette = hot(16)
        elif self.attr == 'unit_density_aa':
            palette = hot(256)
        elif self.attr == 'player_id':
            palette = PLAYER_ABSOLUTE_PALETTE
        elif self.attr == 'player_relative':
            palette = PLAYER_RELATIVE_PALETTE
        elif self.attr == 'selected':
            palette = SELECTED_PALETTE
        elif self.attr == 'effects':
            palette = EFFECTS_PALETTE
        elif self.attr == 'camera':
            palette = CAMERA_PALETTE

        if self.np_data is None:
            self._numpized()
        self.rgb_data = numpy.take(palette, self.np_data, axis=0, mode='clip').astype(numpy.uint8)
