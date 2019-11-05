from sc2.constants import *
from sc2.static_data import UNIT_TYPES

import numpy

_WHAT_UNITS_USE = [
    UnitTypeId.MINERALFIELD,
    UnitTypeId.MINERALFIELD750,
    UnitTypeId.VESPENEGEYSER,

    UnitTypeId.NEXUS,
    UnitTypeId.PYLON,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.GATEWAY,
    UnitTypeId.FORGE,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.ROBOTICSFACILITY,
    UnitTypeId.STARGATE,
    UnitTypeId.TWILIGHTCOUNCIL,
    UnitTypeId.ROBOTICSBAY,
    UnitTypeId.FLEETBEACON,
    UnitTypeId.TEMPLARARCHIVE,
    UnitTypeId.DARKSHRINE,

    UnitTypeId.PROBE,
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.SENTRY,
    UnitTypeId.ADEPT,
    UnitTypeId.MOTHERSHIPCORE,
    UnitTypeId.OBSERVER,
    UnitTypeId.WARPPRISM,
    UnitTypeId.IMMORTAL,
    UnitTypeId.PHOENIX,
    UnitTypeId.VOIDRAY,
    UnitTypeId.ORACLE,
    UnitTypeId.COLOSSUS,
    UnitTypeId.DISRUPTOR,
    UnitTypeId.CARRIER,
    UnitTypeId.MOTHERSHIP,
    UnitTypeId.TEMPEST,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.DARKTEMPLAR
]

_WHAT_UNIT_USE_ONEHOT = []

for i, v in enumerate(_WHAT_UNITS_USE):
    _WHAT_UNIT_USE_ONEHOT.append(numpy.zeros(len(_WHAT_UNITS_USE)).astype(numpy.uint8))
    _WHAT_UNIT_USE_ONEHOT[-1][i] = 1

array_size = max(UNIT_TYPES) + 1
UNIT_TYPE_ONEHOT = numpy.zeros((array_size, len(_WHAT_UNIT_USE_ONEHOT))).astype(numpy.uint8)

for i, v in enumerate(_WHAT_UNITS_USE):
    UNIT_TYPE_ONEHOT[v.value] = _WHAT_UNIT_USE_ONEHOT[i]

# Visibility One-hot
VISIBILITY_ONEHOT = numpy.array([
    [1, 0, 0],  # Hidden
    [0, 1, 0],  # Fogged
    [0, 0, 1],  # Visible
])
# Player-relative  One-hot
PLAYER_RELATIVE_ONEHOT = numpy.array([
    [0, 0, 0, 0],       # Background.
    [1, 0, 0, 0],       # Self. (Green).
    [0, 1, 0, 0],       # Ally.
    [0, 0, 1, 0],       # Neutral. (Cyan.)
    [0, 0, 0, 1],       # Enemy. (Red).
])

_WHAT_EFFECTS_USE = [
    EffectId.PSISTORMPERSISTENT,
    EffectId.GUARDIANSHIELDPERSISTENT,
    EffectId.TEMPORALFIELDGROWINGBUBBLECREATEPERSISTENT,
    EffectId.TEMPORALFIELDAFTERBUBBLECREATEPERSISTENT,
    EffectId.THERMALLANCESFORWARD
]
_WHAT_EFFECTS_USE_ONEHOT = []

for i, v in enumerate(_WHAT_EFFECTS_USE):
    _WHAT_EFFECTS_USE_ONEHOT.append(numpy.zeros(len(_WHAT_EFFECTS_USE)).astype(numpy.uint8))
    _WHAT_EFFECTS_USE_ONEHOT[-1][i] = 1

array_size = 13
EFFECT_ONEHOT = numpy.zeros((array_size, len(_WHAT_EFFECTS_USE_ONEHOT))).astype(numpy.uint8)

for i, v in enumerate(_WHAT_EFFECTS_USE):
    EFFECT_ONEHOT[v.value] = _WHAT_EFFECTS_USE_ONEHOT[i]







