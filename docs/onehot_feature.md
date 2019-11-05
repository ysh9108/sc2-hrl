# One-Hot Feature
기존의 Feature Layer를 one-hot으로 나타낸 기능입니다.  

## UnitType (Minimap, Screen)
원핫의 차원은 사용자가 정의한 N개의 차원으로 나타내지며, N은 사용자가 사용하겠다고 정의한 UnitType의 개수입니다.

원핫이 적용되는 UnitType은 다음을 참고하세요.
```python
# sc2\onehot.py

 _WHAT_UNITS_USE = [
    UnitTypeId.NEXUS,
    UnitTypeId.PYLON,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.GATEWAY,
    ...
```
리스트에 표시된 순서대로 인덱스가 부여됩니다. (예. `Nexus:1`, `Pylon:2`)  
원핫으로 바뀌게 되면 다음과 같이 변환됩니다. (예. `[1, 0, 0, ...]`, `[0, 1, 0, ...]`)  

## Effects (Screen)
Effects에는 여러 종족이 쓰고 있는 효과들이 포함되어 있기 때문에, PvsP에서 사용되는 스킬들을 마스킹 하였습니다.
사용/미사용 되는 스킬들은 다음과 같습니다.

원핫의 차원은 사용하겠다고 선언한 N이며, 프로토스의 효과만 사용하였을 때 shape는 (height, width, 5)입니다.

이름 | 종족 | 사용유무
|:---|:---:|:---:|
PSISTORMPERSISTENT | Protoss | O
GUARDIANSHIELDPERSISTENT | Protoss | O
TEMPORAL FIELD GROWING BUBBLE CREATE PERSISTENT | Protoss  | O
TEMPORAL FIELD AFTER BUBBLE CREATE PERSISTENT | Protoss | O
THERMALLANCESFORWARD | Protoss | O
SCANNERSWEEP | Terran | X
NUKEPERSISTENT | Terran | X
LIBERATOR TARGET MORPH DELAY PERSISTENT | Terran | X
LIBERATOR TARGET MORPH PERSISTENT | Terran | X
BLINDING CLOUD CP | Zerg | X
RAVAGER CORROSIVEBILE CP | Zerg | X
NUKE PERSISTENT | Zerg | X

## Player Relative (MiniMap, Screen)
PlayerRelative의 Self, Ally를 합쳐 총 4개의 feature로 변화되었습니다.  
shape:(width, height, 4) 
```
PLAYER_RELATIVE_ONEHOT = numpy.array([
    [0, 0, 0, 0],       # N/A
    [0, 1, 0, 0],       # Self
    [0, 1, 0, 0],       # Ally
    [0, 0, 1, 0],       # Neutral
    [0, 0, 0, 1],       # Enemy
])
```

## Visibility Map (MiniMap, Screen)
Visibility Map의 FullHidden은 사용하지 않아 총 4개의 feature로 변화되었습니다.  
shape:(width, height, 3) 
```
VISIBILITY_ONEHOT = numpy.array([
    [1, 0, 0],  # Hidden
    [0, 1, 0],  # Fogged
    [0, 0, 1],  # Visible
])
```


## Usage
```python
  # sc.BotAI
  
  async def on_step(self, iteration):
    self.state.feature_layer['screen']['unit_type'].one_hot # ex. (84, 84, N of UnitTypes)
```
