# SD² S1 실험 코드

상태: 2026-09-17 `experiment` 루트로 분리한 뒤 로컬 테스트 25개를 다시 통과했습니다. 실제 Llama 8B 체크포인트를 K2에서 실행한 결과는 아직 없습니다. 로컬에 생성되는 `reports/local_tests.json`은 **작은 무작위 CPU 모델**의 테스트 결과이며 연구 결과가 아닙니다. 이 보고서는 Git에 포함하지 않습니다.

## 1. 작업 루트와 Git 배포

로컬 `Speculative Decoding/experiment`만 VS Code와 Git 저장소의 루트로 사용합니다. 상위 연구 문서 폴더는 업로드하지 않습니다. 코드 전달은 ZIP/SFTP가 아니라 **로컬 수정 → commit/push → Seraph clone/pull**로 통일합니다.

GitHub 저장소: `https://github.com/parameter-space/speculative-decoding-experiment` (사용자 요청으로 공개).

최초 내려받기는 할당된 K2 셸에서 실행합니다. 이미 `/ceph_data/leetj3610/experiment`가 있으면 덮어쓰지 말고 기존 저장소부터 확인합니다.

```bash
cd /ceph_data/leetj3610
git clone https://github.com/parameter-space/speculative-decoding-experiment.git experiment
cd experiment
git log -1 --oneline
```

공개 저장소이므로 위 HTTPS clone과 이후 pull에는 GitHub 로그인이 필요하지 않습니다. 로컬 push에는 소유 계정 인증이 필요합니다. 토큰을 URL, 명령 인자, 코드, 문서 또는 채팅에 넣지 않습니다.

이후 로컬에서 변경 파일을 검토하고 commit/push합니다. 서버에서 해당 코드의 실행이 끝난 뒤 다음으로 업데이트합니다.

```bash
cd /ceph_data/leetj3610/experiment
git status --short
git pull --ff-only
git log -1 --oneline
```

서버에 수정 파일이 있거나 fast-forward가 불가능하면 중단하고 원인을 확인합니다. 강제 덮어쓰기, reset, 자동 stash는 하지 않습니다. 로컬과 서버의 commit을 맞춘 후 실행합니다.

`.gitignore`는 환경·vendor·모델·데이터·결과·로그·인증/접속 설정을 제외합니다. Shell 파일은 `.gitattributes`로 LF 줄바꿈을 유지합니다. 데이터셋은 compute-local SSD, 모델·캐시·결과는 Ceph, Conda는 개인 `/data` 경로를 사용합니다. VS Code Remote SSH는 사용하지 않습니다.

## 2. K2 GPU 2장 할당 안에서 환경 준비

현재 계획은 K2 GPU **총 2장**입니다. 기존 1장 할당이 있다면 실행 중인 작업이 없는지 확인하고 할당 전환 여부를 직접 결정합니다. 스크립트는 기존 Job을 취소하거나 확장하지 않습니다. 새 2장 할당이 필요한 경우 master에서 아래를 실행합니다. 128G는 두 모델 복사본의 로딩을 위한 CPU RAM이며 GPU VRAM 설정이 아닙니다.

```bash
srun --partition=debug_ugrad --account=ugrad --qos=qos_leetj3610_2026_2 \
  --nodelist=ariel-k2 --gres=gpu:high_perf:2 --cpus-per-task=16 \
  --mem=128G --time=02:00:00 --pty bash
```

이후 K2 compute 셸에서:

```bash
cd /ceph_data/leetj3610/experiment
bash scripts/setup_k2.sh
```

- 기존 `my_env`를 수정하지 않고 `sd2_s1` 환경을 별도로 만듭니다.
- Python 3.12, PyTorch 2.5.1+cu121, Transformers 4.52.4를 사용합니다.
- 원본 저장소를 지정 커밋으로 내려받고 수정하지 않습니다.
- Anaconda/Hugging Face 약관을 자동 승인하지 않습니다. 접근 오류가 나면 사용자 계정에서 권한을 확인해야 합니다.
- Hugging Face 인증이 안 되어 있다면 **이 환경을 활성화한 compute 셸에서** `huggingface-cli login`으로 직접 로그인합니다. 토큰은 대화·설정·스크립트에 넣지 않습니다. 두 Meta 모델 접근 권한이 필요합니다.

## 3. 데이터 준비와 첫 실험

K2 실측 경로는 복수형 `/local_datasets`와 `/data2/local_datasets`입니다. 현재 실행은 여유 공간을 확인한 `/data2/local_datasets`의 본인 하위 폴더를 사용합니다. 다른 노드로 이동하면 경로와 권한을 다시 확인합니다.

```bash
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
bash scripts/run_k2.sh
```

처음 실행하면 고정된 공식 데이터셋의 Parquet에서 prompt를 선정하고 불변 manifest를 만듭니다. 데이터셋 스크립트나 생성된 코드는 실행하지 않습니다. 데이터 준비 후 실패했다면 같은 manifest를 재사용하고, 분할 파일을 손으로 수정하지 않습니다. 기존 run 디렉토리는 덮어쓰지 않습니다.

데이터 준비는 한 번만 수행합니다. 이후 Slurm이 배정한 GPU마다 독립 worker가 Target8B+Drafter1B 전체를 올립니다. 각 worker는 같은 두 prompt의 원본/관찰 hook·greedy AR 검사와 calibration32를 수행한 후 자연8개·binding4쌍씩 처리합니다. 합계는 자연16개·binding8쌍 그대로입니다. Batch1, 입력 최대896/생성128, draft block4, 개입은 depth1이며 학습하지 않습니다. Tensor parallel이나 VRAM 통합은 사용하지 않습니다. 한 장만 보이는 할당에서도 단일 worker로 동작합니다.

환경 설치가 끝났고 새 Batch Job이 필요한 경우에만 master의 프로젝트 루트에서 `mkdir -p logs`, `export S1_DATA_DIR=...`, `sbatch scripts/smoke_k2.sbatch`를 사용할 수 있습니다. 위 interactive 작업과 불필요하게 중복 실행하지 않습니다. Dataset은 배치가 실행될 K2 local SSD에 있어야 합니다.

## 4. 결과와 중단 해석

```text
runs/<실행시각>-s1-<job>/
  manifests/environment.json, models.json, data.json, run.json
  tensor_map.json
  trace/boundaries.jsonl
  results/S1_endpoint.csv, S1_cases.md, cases.json
  reports/checkpoint_keys.json, tests.json, HANDOFF.md
  worker-0/, worker-1/             # 독립 검사·측정·로그; 최상위에 병합 결과
```

아주 이른 환경·접근 실패에서는 모델·tensor·boundary 파일이 만들어지지 않을 수 있습니다. 이를 완료 결과로 채우지 않습니다.

병합기는 두 worker의 데이터·모델·calibration manifest 일치와 사례 중복을 검사합니다. 하나가 실패하면 최상위 결과는 성공으로 처리하지 않습니다. 체크포인트 키 검사와 상세 GPU 메모리는 각 worker의 보고서에 있습니다. 두 GPU 병렬 처리 시간을 단일 요청의 추론 속도 향상으로 해석하지 않습니다.

- `complete`: 필요한 사례와 calibration이 모두 끝나고 필수 검사를 통과했습니다. **효과가 양수라는 뜻은 아닙니다.**
- `partial`: 정상 반복 경계가 없는 prompt 등이 있습니다. 누락 사유와 실제 완료 수를 확인합니다.
- `failed`: 정렬/동일성/환경 등의 오류입니다. 기존 숫자 행에도 `run_valid=false`가 표시되며 연구 결과로 사용하면 안 됩니다.
- `matched_donor=NA`: 조건에 맞는 calibration donor가 없습니다. UltraChat-only calibration이므로 수학·코드·요약에 같은 domain donor가 없는 것은 예상된 제한입니다. 다른 domain으로 대체하지 않습니다.
- 모든 snapshot을 GPU에 쌓거나 모델 전체를 복제하지 않습니다. 전체 CPU snapshot 저장은 기본 꺼져 있으며 필요하면 직접 실행 시 `--save-snapshots`를 붙입니다.
- 복사·추가 Target 호출을 포함한 **진단 코드**입니다. 이 실행 시간을 추론 속도 향상으로 보고하지 않습니다.

실행 뒤 `reports/HANDOFF.md`, `reports/tests.json`, `results/S1_endpoint.csv`, `results/S1_cases.md`를 전달하면 실제 결과를 검토할 수 있습니다. 인증 정보가 들어간 터미널 전체 로그는 공유하지 마세요.

## 5. 데이터 분리

UltraChat train_sft 32개는 평균/대체 신호용이며 모델 학습 데이터가 아닙니다. 자연 smoke는 대화·수학·코드·요약 각4개입니다. 각32개 pilot와 최대128개 holdout도 처음부터 겹치지 않게 예약하지만 이번 smoke에서는 실행하지 않습니다. 길이 제한·중복 제거 때문에 holdout이 줄어들면 실제 수를 기록합니다.

Binding 쌍은 내용 연결만 바꾸고 label 빈도·prefix token 길이·공통 bridge·단일 token label을 검사합니다. 자연 생성 중 snapshot과 달리, 합성 데이터는 unguided D prefill을 쓰며 `synthetic_teacher_prefix`로 별도 표시합니다. 이번 8쌍을 S2/S3 일반화 검증으로 주장하지 않습니다.
