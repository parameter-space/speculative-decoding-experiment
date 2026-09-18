# 외부 분석기용 코드 읽기 안내

이 저장소는 SD²의 현재 guide G를 교체해 같은 prefix의 다음-token 분포를 비교하는 depth1 진단 코드다. 현재 완료된 측정은 Job 426156의 **FP64 참조 S1 smoke**이며 기본 저정밀 실행, S2 학습, EAGLE-3 구현 또는 속도 벤치마크가 아니다.

## 읽는 순서

1. `COMPATIBILITY.md`: 기본 경로와 FP64 참조의 차이, 수치 검증의 범위.
2. `configs/smoke.json`: 모델/upstream/data revision, seed, 표본 수, 상한. `precision=official_eval`만 보고 실제 실행 정밀도를 판단하지 않는다.
3. `scripts/s1_fp64_k2.sbatch`: 완료된 측정에 사용하도록 제공한 실행 경로. `--target-fp64-reference`, K2 GPU1, 24시간.
4. `signal_study/run.py`: execution policy, baseline/calibration/endpoint 순서, 조건, 결과 부호, 불완전 run 무효화.
5. `signal_study/capture.py`, `state.py`, `validation.py`: 실제 정상 round 캡처, pending slice·cache 복원, p/q 같은 prefix 비교, CPU FP64 측정.
6. `signal_study/reference_precision.py`, `live_precision.py`, `gpu_attention.py`: Target FP64 선형/norm/RoPE, dtype/autocast 감사, exp/sum attention 및 CPU 대조.
7. `signal_study/data.py`, `common.py`: frozen split, 첫 유효 경계, synthetic teacher prefix, donor matching. 자연 상태와 합성 상태를 혼동하지 않는다.
8. `signal_study/diagnose.py`, `precision.py`, `reference_audit.py`, `cpu_attention.py`, `cpu_recovery.py`: 이전 실패를 분리한 진단 경로. 일부 정책은 의도적으로 과거 실패 경로를 보존한다.
9. `tests/`, `scripts/test_local.py`: 로컬 CPU tiny-model 테스트. 실제 checkpoint GPU 결과와 별개다.

## 결과와 버전 증거

- 원시 결과·데이터·weight·인증·접속 설정은 공개 저장소에 포함하지 않는다. 별도 전달 묶음의 CSV, tests.json, S1_cases.md와 상세 인계 문서를 함께 읽는다.
- 이번 GitHub 커밋은 로컬의 누적 구현을 공개한 검토용 코드 스냅샷이다. 서버는 SFTP로 배포했으므로 현재 commit ID만으로 Job 426156 당시 코드의 byte 일치를 증명하지 않는다.
- 정확한 실행 파일 대조에는 서버 결과의 `manifests/implementation.json`을 사용한다. 이 파일은 `signal_study/*.py` hash를 기록하며 shell scripts나 외부 vendor까지 모두 증명하는 manifest는 아니다.
- 모델/data/runtime 증거는 해당 run의 `manifests/{models,data,run,environment,execution_policy,metric_policy,calibration,shard}.json`, `tensor_map.json`, `trace/boundaries.jsonl`, `reports/checkpoint_keys.json`에 있다. 아직 전달받지 않은 파일의 내용은 추정하지 않는다.
- top5와 aggregate overlap만으로 전체 p/q를 복원할 수 없다. 현재 scalar CSV의 donor-label probability는 상대 쌍의 label을 뜻하며 recipient-label probability와 다르다.

## 분석 시 지켜야 할 구분

`A=sum(min(p,q))`, `delta_A=control-original`, `U=original-control`이다. depth1 overlap을 실제 block acceptance나 속도라고 부르지 않는다. 원래 신호를 평균으로 바꾸는 것은 과거 KV에 남은 Target 정보를 제거하는 실험이 아니다. 자연 16 prompt와 binding 8 pair를 따로 분석하고, matched_donor NA는 0으로 채우지 않는다. binding 양방향은 16개 독립 문제가 아니다.

코드 4개가 모두 도입 문구 끝점에서 측정됐고 binding에는 비라벨 token 경쟁이 컸다는 후속 사례 분석을 고려한다. 이는 실행 오류가 확인됐다는 뜻도, 의미 정보가 전혀 없다는 뜻도 아니다. 효과 측정 성공과 의미 해석 가능성을 별도로 검토한다.

실행·업로드·새 설계 변경은 사용자의 요청 범위에 따른다. 코드 검토만 위해 Seraph에 접속하거나 새 GPU 작업을 제출할 필요는 없다.
