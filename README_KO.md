# SD² S1 실험 코드

상태: FP64 참조 S1 smoke Job **426156**, **COMPLETED / 0:0 / 20분 6초**(2026-09-18 사용자 첨부 확인). 검증 64/64, 효과 측정 32 endpoint·128행(113 ok, 적격 donor 부재 15 NA), 후속 사례 분석까지 완료. 자연 입력 mean/mean_rms는 original 대비 평균 overlap +1.2284/+1.4314%p, binding swap은 -1.7108%p였습니다. 코드 4개는 모두 실제 코드 앞 도입 문구를 측정했고, binding Draft의 original/swap top5에는 라벨이 없으며 문장 부호·키 이름 등도 변했습니다. 따라서 작은 FP64 endpoint 실험이며 원래 BF16 재현·속도·의미 전달의 확증은 아닙니다. 기존 결과·설정을 보존하고 같은 작업을 다시 제출하지 않습니다. 의미 해석을 위한 측정 위치·양쪽 라벨 확률 보완은 별도 후속 과제이며 아직 실행하지 않았습니다.

### 완료된 FP64 참조 S1 효과 측정 — 구현 및 재현 기록

2026-09-18 사용자 진행 요청에 따라 검증된 v3 연산을 S1 측정에 연결했습니다. `signal_study.run --target-fp64-reference`는 별도 정책 `s1-target-fp64-exp-sum-v1`이며 기존 official_eval 기본 경로와 구분합니다. Target FP64 계산·KV, Drafter/guidance BF16, GPU exp/sum attention과 매 호출 CPU 대조를 유지합니다. 원래 BF16 실행 재현이나 속도 결과가 아닙니다. config·데이터·checkpoint·허용치 상한/산정 규칙은 그대로이며 전체 preflight 전용 작업을 다시 제출하지 않습니다. 새 측정 안에서 baseline·각 snapshot의 검증은 계속 수행합니다.

calibration 32개로 평균·donor pool을 만들고, 자연 16개 × 5조건(원래 G/self-copy/평균/RMS 보정 평균/matched donor), binding 16개 × 3조건(원래 G/self-copy/swap)을 측정합니다. CSV는 128행이며 donor 부재는 NA로 남습니다. 진행 중·실패·불완전 run의 행은 `run_valid=False`입니다. 최종 complete 이후에도 조건별 NA는 구분합니다. 실행 정책은 manifest·tests·CSV·HANDOFF에 기록합니다. Job 426156의 실제 결과는 `runs/s1-fp64-426156-G61Q5S/measurement`에 있습니다. 아래 업로드·제출 명령은 재현용이며 현재 재실행할 단계가 아닙니다.

로컬 CPU 테스트 **87개 통과**, 새 batch의 `bash -n` 및 `git diff --check` 통과. 작은 실제 SD² 모델의 FP64 전체 측정(26행), 조건별 정책 표시, self-copy 효과 0, 고의 실패 시 앞선 행 무효화·정밀도 복구·단일 worker 제한을 포함합니다. 실제 8B 효과 결과로 해석하지 않습니다.

SFTP로 변경된 `signal_study/run.py`와 새 `scripts/s1_fp64_k2.sbatch`를 업로드합니다. 이전 Job 426094에 사용한 참조 모듈은 그대로 필요합니다. 실행·대기 Job이 사용하는 코드를 덮어쓰지 않습니다. K2 GPU1·CPU8·RAM64G·24시간, 기존 환경·모델·고정 데이터만 사용하며 다운로드하지 않습니다. master에서 제출합니다:

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sd2_s1_job=$(sbatch --parsable scripts/s1_fp64_k2.sbatch)
echo "S1 작업번호: $sd2_s1_job"
```

종료 후 아래를 한 번에 수집합니다. 새 셸에서는 `sd2_s1_job`을 이번 제출 번호로 다시 지정합니다. `tests.json`의 `complete`와 CSV의 `run_valid`까지 확인하며 Slurm 정상 종료만으로 효과를 주장하지 않습니다.

```bash
sacct -j "$sd2_s1_job" --format=JobID,State%20,ExitCode,Elapsed,Timelimit,NodeList,MaxRSS
tail -n 100 "logs/s1-fp64-${sd2_s1_job}.out" "logs/s1-fp64-${sd2_s1_job}.err"
for s1_report in runs/s1-fp64-"${sd2_s1_job}"-*/measurement/{reports/HANDOFF.md,reports/tests.json,results/S1_endpoint.csv}; do
  if [ -f "$s1_report" ]; then
    printf '\n%s\n' "$s1_report"
    cat "$s1_report"
  fi
done
```

### 완료된 GPU 참조 검증 — 구현 및 재현 기록

426051은 오류를 현재 K2/PyTorch 환경의 **FP64 GPU softmax 경로**로 좁혔습니다. 동일 score에서 길이 513/719/769의 GPU softmax 최대 오차는 약 1이지만, 명시적 `exp(x-max(x))/sum(exp(x-max(x)))`는 CPU와 약 `2.22e-16`로 일치했습니다. QK 오차는 0, 동일 확률 PV는 약 `1e-15` 이내입니다. 실제 모델의 769-token QKV에서도 재현됐고 770에서는 재현되지 않았습니다. 조사한 길이·dtype·환경에 관한 결과이며 모든 홀수 길이, 다른 GPU, BF16/FP32 오류를 같은 원인으로 단정하지 않습니다. 특정 upstream 소스 결함/수정 버전은 아직 미확정입니다.

새 정책 `target-fp64-exp-sum-attention-reference-v3`는 FP64 Target attention을 GPU의 QK → max/exp/sum → PV로 계산합니다. 모델 로드 전에 길이 471/512/513/592/656/719/769/770의 causal/물리 mask/cached 총 24건을 CPU 참조·반복 계산과 `1e-9`로 대조합니다. 실패하면 모델 실험으로 넘어가지 않습니다. 이후 매 Target 호출의 마지막 query와 처음 두 긴 shape 전체 출력을 CPU 참조와 대조하면서 고정 64개 preflight를 실행합니다. 반환하는 값은 GPU 결과이며 CPU는 검사만 합니다. 기존 모델/config/upstream·Drafter·endpoint 상한과 산정 규칙은 유지합니다. v1/v2와 원래 BF16 경로는 보존하고, 별도의 진단 참조로만 사용합니다. S1 효과 측정은 자동 실행하지 않습니다.

로컬 CPU unittest **84개 통과**: 긴 홀수/짝수 길이, causal/물리/전체 mask, cached query, softmax 호출 금지, 고의 오차 검출, 예외 복구, Drafter 불변, 실제 축소 모델의 전체 검증을 포함합니다. **실제 K2 v3도 통과**: 15,488회 CPU 마지막-query 대조 최대 오차 `1.4432899e-14`, 길이 769/770 전체 대조 각각 `4.4408921e-16`; 64건의 restore/repeat/G→Delta/self-copy/A-B-A 오차 모두 0. 결과는 `runs/diagnose-426094-FY6FHY/probe/`에 있습니다. 아래 명령은 재현용 기록이며 지금 다시 제출할 단계가 아닙니다.

SFTP 업로드 4개: `signal_study/diagnose.py`, `signal_study/live_precision.py`, `signal_study/gpu_attention.py`, `scripts/gpu_reference_k2.sbatch`. 이전 업로드의 참조 모듈은 그대로 필요합니다. 실행·대기 중 이 코드를 쓰는 Job이 없을 때 업로드합니다. K2 GPU1·CPU8·RAM64G·24시간, 기존 환경·모델·데이터 캐시를 사용합니다.

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sd2_job_id=$(sbatch --parsable scripts/gpu_reference_k2.sbatch)
echo "작업번호: $sd2_job_id"
```

작업 종료 후 같은 셸에서 다음 출력을 한 번에 수집합니다. 앞 단계 실패로 뒤 파일이 없을 수 있으며 아래 루프는 존재하는 보고서만 출력합니다.

`sd2_job_id`는 이번 `sd2-gpu-ref` 제출에서 받은 번호여야 합니다. 이전 CPU 참조 번호 `426051`로 `gpu-ref` 로그를 조회하면 파일이 없습니다. 번호가 불확실하면 `squeue`/`sacct`의 JobName으로 먼저 찾고, 중복 제출하지 않습니다.

```bash
sacct -j "$sd2_job_id" --format=JobID,State%20,ExitCode,Elapsed,Timelimit,NodeList,MaxRSS
tail -n 120 "logs/gpu-ref-${sd2_job_id}.out" "logs/gpu-ref-${sd2_job_id}.err"
for sd2_report in runs/diagnose-"${sd2_job_id}"-*/probe/{gpu-attention-probe.json,preflight-summary.json}; do
  if [ -f "$sd2_report" ]; then
    printf '\n--- %s ---\n' "$sd2_report"
    cat "$sd2_report"
  fi
done
```

모델 로드 전 실패 등으로 summary가 없으면 같은 run의 `probe/diagnostic.json`에 오류가 남습니다. CPU 참조 통과 결과는 `runs/diagnose-426051-SzSWCT/probe/full/preflight-summary.json`에 보존합니다. 전체 구간의 CPU 마지막-query 검사 15,488회, 최대 오차 `1.5598634e-14`; peak GPU allocated `36,827,345,920` / reserved `48,150,609,920` bytes였습니다.

### 이전 단계: 메모리 제한형 FP64 Target 참조 검증

Job 425862에서 Target FP32 KV와 Drafter BF16 KV가 실제 생성됐음을 확인했습니다. baseline에서 정한 TV 허용치 `1e-6`를 17개가 초과했고(최대 TV `3.2713e-6`), 다른 1개는 최대 logit 오차 한도 `4.4823e-5`를 초과했습니다. 이는 새 baseline에서 고정된 한도를 넘은 결과이며, 사후 한도 상향으로 통과 처리하지 않습니다.

`target-fp64-chunked-reference-v1`은 Target의 선형 연산·RMSNorm·RoPE·hidden·실제 생성 KV·출력층을 FP64로 계산하는 **별도의 느린 수치 검증 경로**입니다. Target 가중치는 FP32로 보관하고 선형 연산 시 출력축 1,024행씩만 FP64로 임시 변환합니다. 전체 FP64 가중치 복사본을 GPU에 올리지 않습니다. Drafter와 guidance는 BF16이며 guidance에 들어가는 FP64 hidden은 먼저 FP32로 변환합니다. math SDPA·TF32 비활성화 및 기존 허용치 산정 규칙을 유지하고, 참조 baseline에서 고정한 뒤 동일 64개 입력을 검증합니다. 기존 BF16/FP32 KV를 형변환해 재사용하지 않습니다.

도입 시 로컬 unittest **63개 통과**: 작은 실제 SD² 모델의 생성·FP64 KV, BF16 Drafter, 8개 축소 입력의 전체 흐름, 분할 선형 연산과 dense FP64 비교, norm/RoPE 정밀도, 예외 후 설정 복구를 포함합니다. **실제 8B/K2 Job 425885는 60/64 통과·4/64 실패**이며, GPU peak allocated 34.40 GiB/reserved 42.92 GiB로 해당 검사는 OOM 없이 끝났습니다. 아래는 전체 참조 검증의 재실행 안내이며 현재 후속 단계는 위 실패 연산 대조입니다. 참조 통과도 기존 정밀도 본 실험의 통과나 속도 개선을 뜻하지 않습니다. config·upstream·기존 본 실험 실행 경로는 변경하지 않습니다.

실행·대기 Job이 해당 코드를 사용하지 않을 때 다음 5개 파일을 SFTP 업로드합니다: `signal_study/diagnose.py`, `signal_study/live_precision.py`, `signal_study/reference_precision.py`, `signal_study/validation.py`, `scripts/reference_k2.sbatch`.

master에서 제출합니다. 기존 데이터·모델 캐시만 사용하며 K2 GPU1·CPU8·RAM64G·최대 24시간입니다.

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sd2_job_id=$(sbatch --parsable scripts/reference_k2.sbatch)
echo "작업번호: $sd2_job_id"
```

작업 종료 후 같은 셸에서 아래 출력을 한 번에 수집합니다. 셸을 새로 열었다면 `sd2_job_id`를 제출 때 출력된 번호로 다시 지정합니다.

```bash
sacct -j "$sd2_job_id" --format=JobID,State%20,ExitCode,Elapsed,Timelimit,NodeList,MaxRSS
tail -n 200 "logs/reference-${sd2_job_id}.out" "logs/reference-${sd2_job_id}.err"
cat runs/diagnose-"${sd2_job_id}"-*/probe/preflight-summary.json
```

요약에는 통과/실패 수, 실제 KV dtype, 최대 TV, 분할 연산 임시 가중치 크기, GPU peak 메모리가 기록됩니다. 전체 세부 결과는 같은 폴더의 `diagnostic.json`입니다. summary 생성 전 로딩 실패나 Slurm 강제 종료가 발생하면 파일이 없을 수 있으므로 Slurm 상태와 로그도 함께 확인합니다.

### 실제 생성부터 Target FP32로 실행하는 사전검증

이전 단계의 재실행 안내입니다. 실제 8B 결과는 Job 425862에서 46/64 통과·18/64 실패로 확인했으며, 최신 후속 단계는 위 FP64 참조 검증입니다.

SFTP 업로드: `signal_study/diagnose.py`, `signal_study/live_precision.py`, `scripts/live_fp32_k2.sbatch`. 실행·대기 Job이 없는 상태에서 업로드한 뒤 master에서 제출합니다.

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sbatch scripts/live_fp32_k2.sbatch
```

K2 GPU1·CPU8·RAM64G·최대 24시간. Target은 생성 시작부터 decoder/head/KV 모두 FP32, Drafter·guidance는 BF16으로 분리합니다. 캡처된 BF16 KV를 FP32로 변환하는 방식이 아닙니다. Target 내부에 등록된 guidance도 별도 BF16 경계를 유지합니다. math SDPA·TF32 비활성화로 두 baseline을 먼저 검사하고, 동일한 허용치 산정 규칙을 적용한 뒤 값을 고정합니다. 통과하면 calibration 32개, 자연 smoke 16개, binding 16개를 검사합니다. endpoint validation 실패는 한 번에 수집하며 허용치를 변경하지 않습니다. baseline 실패나 예상하지 못한 실행 오류는 중단합니다.

결과는 `runs/diagnose-<JOB>-<unique>/probe/preflight-summary.json` 및 `diagnostic.json`입니다. `logs/live-fp32-<JOB>.out/.err`에 진행 및 오류가 남습니다. 실패가 있으면 종료 코드 2, 모두 통과하면 0입니다. 통과하더라도 S1 효과 측정·전체 연구 실험 완료가 아닙니다. 기존 `official_eval` 본 실험은 그대로이며 새 정책 ID `target-fp32-live-preflight-v1`을 별도로 기록합니다. FP32 전환으로 생성 경로가 달라질 수 있으므로 이전 BF16 결과와 섞지 않습니다.

### Target 전체 FP32 비교 진단 (2026-09-18)

해당 재계산 진단은 Job 425852에서 실제 8B 실행까지 완료됐습니다. 아래는 과거 단계의 재실행 안내이며 최신 후속 단계는 위 FP64 참조 검증입니다.

`signal_study/diagnose.py`, 새 `signal_study/precision.py`, 새 `scripts/precision_k2.sbatch`를 SFTP 업로드합니다. 기존 Job 종료 후 master에서 아래를 제출합니다.

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sbatch scripts/precision_k2.sbatch
```

K2 GPU 1장·CPU8·RAM64G·최대 24시간입니다. 기존 정밀도의 baseline과 동일 calibration 입력을 재현해 native gate 결과를 보존한 뒤, autocast 범위를 나와 Target의 저장된 weight 값을 FP32로 승격합니다. 이는 잃어버린 원본 정밀도 복원이 아닙니다. autocast/TF32를 끄고 math SDPA로 논리/물리 문맥의 full/split을 각각 새 FP32 KV로 계산합니다. 기존 BF16 KV를 형변환해 쓰지 않습니다. 원래 KV 생성 이력까지 재현하는 검사는 아니므로 FP32 차이가 작아져도 모든 캐시 버그를 배제하지 않습니다.

`logs/precision-<job>.out/.err`와 출력된 run의 `probe/diagnostic.json`에서 `precision_probe.native_gate`, `native_alignment`, `fp32`를 함께 봅니다. 종료 코드 0과 `Precision diagnostic complete`는 비교 계산 완료이지 native gate/S1 통과가 아닙니다. 기존 본 실험의 정밀도·허용치·핀은 변경하지 않습니다. Job 425852의 제한적 재계산은 K2 한 장에서 OOM 없이 완료했지만 더 긴 실제 생성의 메모리 적합성까지 보장하지 않습니다.

### Calibration 확률 합 오류 수정 및 재검증

측정용 softmax·확률 합·TV·overlap을 CPU FP64로 계산합니다. 모델 weights/forward, upstream sampling, 합 오차 상한 `1e-5`, baseline 상한 및 tolerance 결정 규칙은 바꾸지 않습니다. 강제 재정규화는 하지 않으며, 여러 토큰/배치를 하나의 분포로 합치던 reshape도 차단합니다. 새 실행은 `cpu-fp64-probability-v1`을 manifest·검사 보고서·결과 행에 기록하고 baseline부터 새로 검증합니다. 이전 FP32 결과와 섞지 않습니다.

원격 실행·대기 Job이 없는지 확인하고 변경 파일을 SFTP 업로드한 뒤, **master에서 제출만** 합니다. 아래 작업은 K2 GPU 1장, CPU8, RAM64G, 최대 24시간이며 기존 캐시만 사용합니다.

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sbatch scripts/probability_k2.sbatch
```

두 baseline과 정확히 실패했던 calibration 입력 하나를 검사합니다. Target fresh/cached/source 및 Drafter original logits의 shape/dtype, FP32/FP64 확률 합과 오차를 `runs/diagnose-<job>-<unique>/probe/diagnostic.json`에 기록합니다. 통과는 해당 입력의 회귀 검사 통과일 뿐이며, 전체 calibration·binding·S1 성공을 뜻하지 않습니다. 이 결과를 검토한 뒤 별도의 `scripts/smoke_k2.sbatch`를 제출합니다. 합성 입력만으로 실제 서버 오류의 원인을 확정하지 않습니다.

2026-09-18 추가 진단: `signal_study/diagnose.py`와 `signal_study/validation.py`를 함께 업로드한 뒤 같은 batch를 제출합니다. TV 실패 메시지에 실제 TV·허용치·logit 오차·argmax·문맥/캐시 길이를 출력하고 실패 전 `checks`도 보존합니다. `alignment_audit`에는 active token/position/mask 및 cache layer 길이 정합성, fresh/cached 반복, 동일 문맥의 새 clean cache 분할 계산, 물리 슬롯·마스크를 유지한 전체 재계산, 동일 hidden의 FP32 출력층 비교를 기록합니다. 구조 일치만으로 KV 내용의 정확성을 입증하지는 않습니다. 추가 비교는 원래 검사가 실행된 뒤의 진단 전용이며 tolerance를 다시 맞추거나 모델·sampling을 수정하지 않습니다. 추가 진단 자체가 실패해도 기존 검사 오류를 덮어쓰지 않습니다. `alignment_audit.status=complete`는 비교 기록 완료이지 endpoint 통과가 아닙니다.

### Baseline 실패 원인 진단

기존 GPU 작업이 끝나 셸 프롬프트로 돌아오면 아래 §1에 따라 변경 코드를 업로드한 뒤, 현재 Slurm 할당의 K2 셸에서 실행합니다. 두 baseline 입력만 사용하며 본 실험으로 이어지지 않습니다.

```bash
cd /ceph_data/leetj3610/experiment
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
bash scripts/diagnose_k2.sh
```

Slurm에 보이는 첫 GPU 한 장과 이미 다운로드한 캐시만 사용합니다. 캐시가 없으면 다운로드하지 않고 중단합니다. `default`, `explicit`, `math` 각각에서 full-prefix/cached-tail과 같은 경로의 반복 결과를 비교하고, 동일 hidden의 출력층만 FP32로 다시 계산한 수치도 기록합니다. FP32 출력층은 저장된 weight를 변환하며 원본의 더 높은 정밀도 weight를 복원하는 것이 아닙니다. 기존 검사 허용치·실험 설정은 변경하지 않습니다.

출력은 `runs/diagnose-<job>-<unique>/console.log` 및 `probe/diagnostic.json`에 남습니다. `Diagnostic complete`는 진단 완료이지 S1 검증 통과가 아닙니다.

2026-09-17 K2의 두 입력 진단에서 math SDPA의 TV 및 최대 logit 오차는 기존 상한 이내였습니다. 다음 명령은 math SDPA를 생성·AR·clean-prefix 계산 전체에 일관되게 적용하여 baseline만 검사하고 종료합니다. 실패하면 종료 코드 2이며, 통과해도 calibration/endpoint 실험은 실행하지 않습니다. 본 실행의 기본 backend나 config는 변경하지 않습니다.

```bash
bash scripts/diagnose_k2.sh --math-baseline-only
```

2026-09-17 사용자 K2 로그에서 math baseline의 두 입력 모두 hook identity, 32-token greedy AR identity, clean-prefix logit/TV 검사를 통과했습니다. 다음 단계는 각 smoke domain의 첫 입력 하나씩(총 4개)을 고정 선택한 snapshot 사전 검증입니다.

```bash
bash scripts/diagnose_k2.sh --math-preflight
```

Baseline에서 정한 tolerance를 그대로 사용해 캐시 복원·반복·self-copy·G→Delta·fresh/cached Target·A-B-A를 검사합니다. 경계가 없거나 검사에 실패하면 실패로 종료하며 쉬운 다른 입력으로 교체하지 않습니다. 이 명령은 전체 calibration, binding, S1 효과 측정을 수행하지 않습니다. 성공해도 전체 S1 실행의 사례별 검사는 계속 필요합니다.

## 1. 작업 루트와 코드 업로드

로컬 `Speculative Decoding/experiment`만 VS Code와 Git 저장소의 루트로 사용합니다. 상위 연구 문서 폴더는 업로드하지 않습니다. 2026-09-17 최신 사용자 요구에 따라 **로컬 VS Code 수정 → SFTP 확장으로 변경 코드 업로드 → Seraph 실행**을 기본으로 사용합니다. ZIP 전달은 사용하지 않습니다.

### 기본: VS Code SFTP 확장

- 로컬 루트는 `experiment/`, 원격 루트는 `/ceph_data/leetj3610/experiment`입니다. VS Code는 로컬 폴더를 열며 Remote SSH는 사용하지 않습니다.
- 실행·대기 중인 실험이 해당 코드를 참조하지 않을 때 변경 파일을 명시적으로 업로드합니다. 기본적으로 저장 시 자동 업로드와 원격 파일 삭제 동기화는 켜지 않습니다. 서버의 결과·캐시·vendor를 덮어쓰거나 지우지 않습니다.
- SFTP 제외 목록을 별도로 설정합니다. `.gitignore`가 SFTP에도 자동 적용된다고 가정하지 않습니다. `.git/`, `.vscode/`, `.idea/`, 환경, `vendor/`, 데이터, 모델, 캐시, `runs/`, `logs/`, `reports/`, `tmp/`, `out/`, 인증 파일은 제외합니다. 세부 제외 대상은 `.gitignore`도 참고합니다.
- 접속 설정은 로컬 전용이며 Git·서버 업로드·문서·채팅에 포함하지 않습니다. shell 파일은 VS Code에서 LF 줄바꿈을 유지합니다. `.gitattributes`만으로 SFTP 전송 시 줄바꿈이 변환되지는 않습니다.
- Git은 버전 관리에 계속 사용할 수 있으나 매번 push/pull할 필요는 없습니다. 업로드 파일 목록과 실행 시 코드 상태를 확인하며, SFTP 수정 후의 서버 Git HEAD만으로 실제 코드 버전이 같다고 판단하지 않습니다.

2026-09-18 사용자가 SFTP 연결 성공을 알렸습니다. 이번 수정 파일의 업로드 성공은 아직 확인하지 않았습니다.

### 대안: VS Code를 사용하지 않을 때 Git

GitHub 저장소: `https://github.com/parameter-space/speculative-decoding-experiment` (사용자 요청으로 공개).

최초 내려받기는 할당된 K2 셸에서 실행합니다. 이미 `/ceph_data/leetj3610/experiment`가 있으면 덮어쓰지 말고 기존 저장소부터 확인합니다.

```bash
cd /ceph_data/leetj3610
git clone https://github.com/parameter-space/speculative-decoding-experiment.git experiment
cd experiment
git log -1 --oneline
```

공개 저장소이므로 위 HTTPS clone과 이후 pull에는 GitHub 로그인이 필요하지 않습니다. 로컬 push에는 소유 계정 인증이 필요합니다. 토큰을 URL, 명령 인자, 코드, 문서 또는 채팅에 넣지 않습니다.

Git 배포를 선택한 경우 로컬에서 변경 파일을 검토하고 commit/push합니다. 서버에서 해당 코드를 참조하는 실행·대기 작업이 없을 때 다음으로 업데이트합니다.

```bash
cd /ceph_data/leetj3610/experiment
git status --short
git pull --ff-only
git log -1 --oneline
```

SFTP 업로드로 서버에 Git 미커밋 변경이 생길 수 있습니다. 수정 파일이 있거나 fast-forward가 불가능하면 중단하고 로컬·서버 차이를 확인합니다. 강제 덮어쓰기, reset, 자동 stash는 하지 않습니다. Git 배포에서는 commit뿐 아니라 미커밋 변경까지 확인한 후 실행합니다.

`.gitignore`는 환경·vendor·모델·데이터·결과·로그·인증/접속 설정을 제외합니다. Shell 파일은 `.gitattributes`로 LF 줄바꿈을 유지합니다. 데이터셋은 compute-local SSD, 모델·캐시·결과는 Ceph, Conda는 개인 `/data` 경로를 사용합니다. VS Code Remote SSH는 사용하지 않습니다.

## 2. K2 GPU 2장 할당 안에서 환경 준비

현재 계획은 K2 GPU **총 2장**입니다. 기존 1장 할당이 있다면 실행 중인 작업이 없는지 확인하고 할당 전환 여부를 직접 결정합니다. 스크립트는 기존 Job을 취소하거나 확장하지 않습니다. 새 2장 할당이 필요한 경우 master에서 아래를 실행합니다. 128G는 두 모델 복사본의 로딩을 위한 CPU RAM이며 GPU VRAM 설정이 아닙니다.

```bash
srun --partition=debug_ugrad --account=ugrad --qos=qos_leetj3610_2026_2 \
  --nodelist=ariel-k2 --gres=gpu:high_perf:2 --cpus-per-task=16 \
  --mem=128G --time=02:00:00 --pty bash
```

위 2시간은 짧은 대화형 환경 점검용 예시입니다. 본 실험의 `sbatch` 기본값은 아래처럼 24시간이며, 최대 4시간인 `debug_ugrad`에 24시간을 요청하지 않습니다.

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
export S1_SDPA_BACKEND=math
bash scripts/run_k2.sh
```

처음 실행하면 고정된 공식 데이터셋의 Parquet에서 prompt를 선정하고 불변 manifest를 만듭니다. 데이터셋 스크립트나 생성된 코드는 실행하지 않습니다. 데이터 준비 후 실패했다면 같은 manifest를 재사용하고, 분할 파일을 손으로 수정하지 않습니다. 기존 run 디렉토리는 덮어쓰지 않습니다.

데이터 준비는 한 번만 수행합니다. 이후 Slurm이 배정한 GPU마다 독립 worker가 Target8B+Drafter1B 전체를 올립니다. 각 worker는 같은 두 prompt의 원본/관찰 hook·greedy AR 검사와 calibration32를 수행한 후 자연8개·binding4쌍씩 처리합니다. 합계는 자연16개·binding8쌍 그대로입니다. Batch1, 입력 최대896/생성128, draft block4, 개입은 depth1이며 학습하지 않습니다. Tensor parallel이나 VRAM 통합은 사용하지 않습니다. 한 장만 보이는 할당에서도 단일 worker로 동작합니다.

사전 검증이 끝나면 interactive 셸에서 `exit`로 할당을 반환하고 master에서 아래 명령으로 S1 smoke를 제출합니다. master에서는 모델이나 Python을 실행하지 않습니다. 이미 제출했다면 중복 제출하지 않습니다.

```bash
cd /ceph_data/leetj3610/experiment
mkdir -p logs
export S1_DATA_DIR=/data2/local_datasets/leetj3610/sd2_s1_smoke_v1
sbatch scripts/smoke_k2.sbatch
squeue -u "$USER" -o "%.18i %.9T %.20N %.12L %R"
```

Batch 요청은 `batch_ugrad`, K2 GPU 2장, CPU 16개, RAM 128G, **기본 제한 24시간(`--time=1-00:00:00`)**입니다. 이는 사용자 작업 기본값이며 학교의 의무 시간이 아닙니다. 별도 합의 없이 smoke라는 이유로 짧게 줄이지 않습니다. 계산이 먼저 끝나면 즉시 종료하며 24시간을 채우기 위해 자원을 유지하지 않습니다. 파일 수정은 이미 제출된 Job의 제한 시간을 바꾸지 않습니다.

기존 모델·데이터 캐시만 사용하며 math SDPA를 baseline·calibration·모든 생성/endpoint 계산에 일관되게 적용합니다. `S1_SDPA_BACKEND=math`는 worker 모델 manifest와 검사 보고서에 기록되어 다른 backend 결과와 섞이지 않습니다. config/data hash, 모델 weight dtype, 오차 상한은 변경하지 않았습니다. 이 설정의 결과로 기본 fused SDPA 속도를 주장하지 않습니다. 실행·대기 중 해당 코드를 참조하는 Job이 있으면 SFTP 업로드·pull·수정을 하지 않습니다.

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
