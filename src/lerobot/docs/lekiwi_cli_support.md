## LeKiwi용 `lerobot-record` CLI 지원에 필요한 수정 사항

LeKiwi는 베이스/팔 제어를 서로 다른 테레옵 두 개(키보드 + SO100/101 리더)로 분리하고, 카메라 스트림도 Jetson 호스트에서만 열립니다. 현재 CLI는 단일 테레옵과 로컬 카메라만 가정하고 있어서 그대로는 사용할 수 없습니다. 아래 수정들을 적용해야 합니다.

1. **LeKiwiClient에 카메라 메타데이터 노출**
   - `robots/lekiwi/lekiwi_client.py`에 `cameras` 속성(또는 프로퍼티)을 추가해 `self.config.cameras` 정보를 그대로 제공하거나, 이미지 writer가 호출할 수 있는 더미 카메라 래퍼를 제공합니다.
   - 이렇게 해야 `len(robot.cameras)` 같은 기존 코드가 깨지지 않고, 비디오 인코딩 설정에 필요한 해상도 정보도 얻을 수 있습니다.

2. **원격 스트림 기반으로 이미지 writer 설정**
   - `scripts/lerobot_record.py`에서 `dataset.start_image_writer`나 `LeRobotDataset.create(... image_writer_threads=...)` 호출 시 `robot.cameras`가 없을 수도 있다는 전제를 추가합니다.
   - 카메라 개수/해상도는 `robot.observation_features`나 `robot.config.cameras`에서 안전하게 읽어오도록 분기 로직을 넣습니다.

3. **멀티 테레옵 지원**
   - `RecordConfig`와 생성/검증 로직을 확장해 `cfg.teleop`가 `Teleoperator` 리스트일 때도 처리하도록 합니다.
   - LeKiwi인 경우에는 팔 리더(`so100_leader`/`so101_leader`)와 키보드 테레옵(예: `KeyboardTeleop`) 두 개가 모두 존재하는지 체크하고, 하나라도 빠지면 명확한 에러 메시지를 출력해야 합니다.
   - `teleop.connect()`, `record_loop(...)` 호출부도 리스트를 그대로 전달하도록 바꿉니다.

4. **CLI 인자 확장**
   - 동일 스크립트에서 `--teleop` 블록을 두 번 넘길 수 있게 하거나, JSON/리스트 형태로 arm + keyboard 설정을 받아 `cfg.teleop` 리스트로 변환하는 파서 헬퍼가 필요합니다.
   - 사용자가 `--teleop[0].type=so100_leader`, `--teleop[1].type=keyboard`처럼 지정할 수 있게 만드는 것이 가장 명확합니다.

5. **문서/에러 메시지 정리**
   - 위 변경을 적용하면 `lerobot-record` 도움말에 “LeKiwi는 teleop arm + keyboard가 모두 필요하다”는 안내와, 예시 명령어를 추가해주는 것이 좋습니다.

### 참고 예제
- 현재 동작하는 LeKiwi 수집 파이프라인 예제는 `examples/lekiwi/record.py`에 있습니다. CLI를 손보기 전에 구조와 요구 조건을 이해하려면 이 파일을 먼저 확인하세요.
