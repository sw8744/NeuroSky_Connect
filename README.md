# NeuroSky_Connect: MindWave Mobile 2를 이용한 졸음 뇌파 구분
# 개요
NeuroSky MindWave Mobile 2를 이용하여 측정된 졸음 뇌파를 분석해 
사용자의 현재 상태가 정상 상태인지, 졸음 상태인지 구분하는 프로젝트이다.


# 필요 프로그램
- [jNeuroSkyAPI](https://github.com/sw8744/jNeuroSkyAPI)에서 Release 되어있는 `jNeuroSkyAPI.jar`
- [NeuroSky ThinkGear Connector](https://developer.neurosky.com/docs/doku.php?id=thinkgear_connector_tgc)

# 사용 방법
1. `client.py`를 실행한다.
2. EEG 측정 여부, Arduino 연결 여부를 선택한다.
3. 만일 EEG를 측정하고 싶은 경우, ThinkGear Connector를 실행하고, `jNeuroSkyAPI.jar`를 실행한다.
4. 측정된 뇌파 데이터를 분석하여 사용자의 상태를 판단할 수 있다.

# 성능
- 현재 정확도는 약 85% 전후로 나온다.
- 추후 개선을 통해 더 정확도를 올릴 수 있을 것이라 예측된다.

