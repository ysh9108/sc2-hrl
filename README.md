# CILAB-sc2

## Quick Start
기본 봇(base.py)을 실행시키기 위한 환경구축을 설명합니다.  


1.  [s2client-proto](https://github.com/Blizzard/s2client-proto)와 [CILAB-sc2](https://github.com/cilab-matser/CILAB-sc2) repository를 clone 하세요.
```
git clone https://github.com/Blizzard/s2client-proto
git clone https://github.com/cilab-matser/CILAB-sc2
```

2. 리눅스 환경 스타크래프트 바이너리 클라이언트를 다운받고, `home\StarCraftII`에다가 압축을 해제하세요.<br>(Password : iagreetotheeula)<br>
이 버전에서는 4.1.2버전을 사용하였습니다.
[S2 Binary](https://github.com/Blizzard/s2client-proto#linux-packages)

```
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
```
3. 다운받은 s2client-proto를 설치합니다.
```
bash s2client-proto\install_protoc.sh
pip install ./s2client-proto
```
4. 다운받은 CILAB-sc2를 설치합니다. <br>
**주의 : 이 프로젝트에서 사용하는 sc2 패키지는 [Dentosal/python-sc2](https://github.com/Dentosal/python-sc2)의 프로젝트의 `sc2`와 다릅니다.<br>따라서 이 프로젝트 패키지가 업데이트 되면 아래 명령어를 다시 실행하여야 합니다.**

```
pip install ./CILAB-sc2
```

### 패키지만 업데이트 하는법
`sc2` 폴더 내의 모든 파일을 [master-branch](https://github.com/cilab-matser/CILAB-sc2/tree/master)에 올려져 있는 것으로 대체한 후,  
```
pip install ./CILAB-sc2
```
명령어를 이용해 패키지를 재설치하세요.

## Document
- [Feature Layer](docs/feature.md)
- [One-Hot Feature](docs/onehot_feature.md)
- [Camera](docs/camera.md)

