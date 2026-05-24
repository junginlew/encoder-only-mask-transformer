# EoMT Mask Annealing — 학습 설정 이슈 분석·검증·수정 기록

## TL;DR

- **코드 버그가 아니다.** `eomt.py`, `base.py`, `lr_scheduler.py`의 mask annealing / LLRD / masked attention 구현 로직 자체는 정확하다.
- **문제는 학습 설정에 있었다.** annealing이 완료되기 전 시점의 체크포인트를 `masked_attn_enabled=False`로 export하면, "학습한 함수(masked)"와 "배포한 함수(maskless)"가 달라져 train/inference gap이 생긴다.
- **근본 원인은 step 회계 불일치.** `accumulate_grad_batches=32` 때문에 옵티마이저 step이 에폭당 약 4개뿐인데, annealing `end_steps`는 옵티마이저 step 단위라서 실제 학습 길이 대비 너무 크게 잡혀 있었다.
- **실측으로 진단 확정.** export 대상 체크포인트(epoch 394, `global_step=1580`)에서 `attn_mask_probs = [0, 0, 0, 0.2455]` — deepest 블록이 아직 24.6% masked. 예측값(0.25)과 소수점 넷째 자리까지 일치.
- **적용한 수정.** `eomt-vitb.yaml`의 `attn_mask_annealing_end_steps`를 `[500,1000,1500,2000]` → `[200, 450, 700, 950]`으로 당김(S=1580 기준 `[0.15S, 0.30S, 0.45S, 0.60S]`).
- **범위 결정.** ViT-L은 배포 대상이 아니므로 config 수정 불필요(스킵). `base.py` 검증 토글(2순위)은 필수가 아닌 보험. 기존 ONNX는 재학습 전까지 gap이 남아 있으므로, 재학습 전 §6.2로 실제 손해부터 측정.

---

## 1. 배경: masked attention과 mask annealing이 무엇인가

EoMT는 별도 디코더 없이, 학습 가능한 쿼리 토큰을 ViT의 마지막 L2 블록들에 끼워 넣어 segmentation을 수행한다. 이때 학습을 돕기 위해 **masked attention**을 쓴다.

- 각 L2 블록 진입 전, 모델이 자기 예측 마스크(`mask_logits`)를 만든다.
- 그 마스크가 양수인 패치에만 쿼리가 attend 하도록 어텐션을 제한한다.
  (`src/models/eomt.py` → `_attn_mask()`, L162–178; `attn_mask[:, :num_q, num_q+num_prefix:] = (interpolated > 0)`)
- 학습이 진행되면 `attn_mask_probs[i]`를 1.0 → 0.0으로 낮춰 제한을 서서히 푼다.
  (`_disable_attn_mask()`, L123–131: `prob=1`이면 완전 masked, `prob=0`이면 마스크를 전부 열어 unmasked와 동일)
- 이 감쇠 스케줄을 적용하는 곳: `src/models/base.py` → `on_train_batch_end()`, L664–699 (블록별 `start/end_steps`로 `prob = (1 - progress)^poly_power` 계산).

추론 시에는 masked attention을 끈다(`export_eomt.py` L30, `masked_attn_enabled = False`). 동적 boolean 마스크는 Hailo DFC가 컴파일할 수 없고, 애초에 학습 보조 장치이므로 떼는 것이 EoMT의 설계 의도다.

---

## 2. 문제 정의: "최적"은 forward 함수에 종속된다

같은 가중치 θ라도 mask를 켠 forward와 끈 forward는 **서로 다른 함수**다.

- `f_masked(x; θ)` — 쿼리가 자기 예측 마스크 영역의 패치만 attend
- `f_unmasked(x; θ)` — 쿼리가 전체 패치를 attend

학습 시 gradient는 `f_masked`의 계산 그래프를 타고 흐른다. 따라서 학습으로 찾은 θ\*는 **`f_masked`를 최적으로 만드는 가중치**이지, `f_unmasked`를 최적으로 만드는 가중치가 아니다. masked attention은 dropout/BatchNorm처럼 기댓값이 보존되는 장치가 아니라 어텐션 연결 자체를 하드하게 끊으므로, 떼면 진짜로 다른 함수가 된다.

**경험적 증거 (논문 Table 7):** annealing 없이 학습한 뒤 추론에서 마스크를 떼면 PQ가 3.0 하락하고, annealing을 적용하면 0.2만 하락한다. "masked 최적 θ"가 "maskless 최적 θ"와 같다면 annealing 여부와 무관하게 낙폭이 ≈0이어야 한다. 3.0의 낙폭은 둘이 다르다는 직접 증거다.

**핵심 성질:** annealing이 모든 블록에서 `prob=0`에 도달하면, 그 이후 `f_masked`는 `f_unmasked`와 **완전히 동일한 함수**가 된다. 따라서 annealing이 best 체크포인트보다 앞서 끝나면 train/inference gap은 원천적으로 사라진다.

개념적으로는 시퀀스 모델의 **teacher forcing / exposure bias**와 같은 구조다. 보조 조건에서만 학습하면 보조 없는 추론에서 손해를 보고, scheduled sampling으로 보조를 서서히 줄여 해소한다. mask annealing이 정확히 그 처방이다. (단, EoMT의 마스크는 GT가 아니라 모델 자기 예측 기반이라는 차이는 있다.)

---

## 3. 근본 원인: 옵티마이저 step 회계

- `configs/experiment/eomt/eomt-vitb.yaml`: `accumulate_grad_batches: 32`, `train_batch_size: 2` → effective batch 64.
- annealing `end_steps`와 `warmup_steps`는 모두 **옵티마이저 step 단위**다 (`on_train_batch_end`에서 `self.global_step` 사용). `global_step`은 배치가 아니라 옵티마이저 step마다 증가한다.
- 실측 결과 에폭당 옵티마이저 step ≈ 4개 (`global_step 1580 / epoch 394 ≈ 4.0`). 즉 한 epoch에 step이 4개씩만 쌓인다.
- 논문의 `max_steps=160,000`은 ADE20K(약 2만 장, 에폭당 step 수가 훨씬 많음) 기준이다. 소규모 데이터(약 400장)에 그대로 쓰면 에폭당 step이 극히 적어 스케줄이 의도대로 진행되지 않는다.

---

## 4. 검증 결과 (실측으로 확정)

export 대상 체크포인트(`scripts/export_eomt.py` L6이 가리키는 `epoch=394`)를 §6.1 스크립트로 직접 열어 버퍼 값을 확인했다.

```
model.attn_mask_probs: [0.0000, 0.0000, 0.0000, 0.2455]
global_step: 1580
epoch: 394
```

| 블록 | end_step(기존) | prob (실측) | 상태 |
|------|----------------|-------------|------|
| block 0 | 500  | 0.0000 | 완전히 풀림 |
| block 1 | 1000 | 0.0000 | 완전히 풀림 |
| block 2 | 1500 | 0.0000 | 완전히 풀림 (step 1580 > 1500으로 막 통과) |
| block 3 | 2000 | **0.2455** | **아직 24.6% 의존 중 (미완)** |

**예측과 실측 일치 확인.** block 3은 `end=2000`, 현재 `step=1580`이므로:

```
progress = 1580 / 2000 = 0.79
prob     = (1 - 0.79) ^ 0.9 = 0.21 ^ 0.9 ≈ 0.2454
```

이론값 0.2454 ≈ 실측값 0.2455 (소수점 넷째 자리까지 일치). 즉 "deepest 블록이 export 시점에 24.6% masked인 상태로 maskless export되어, 그 블록에 train/inference gap이 있었다"는 진단이 **추정이 아니라 확정**됐다. (step/epoch가 ~4로 측정돼 초기 추정 ~3보다 약간 많았고, 그래서 best가 step 1580까지 도달한 것과도 일관된다.)

---

## 5. 적용한 수정

### 5.1 (적용 완료) ViT-B `end_steps` 재캘리브레이션

annealing이 모든 블록에서 `prob=0`에 도달하면 그 이후 masked == maskless가 되므로, **deepest end_step이 best 체크포인트 step보다 충분히 작아지도록** 당긴다. 단 너무 급하면 초반 수렴 도움이 줄어드니, best step의 약 40~60% 지점에서 끝나도록 잡는다.

실측 `S = global_step = 1580`에 `[0.15S, 0.30S, 0.45S, 0.60S]` 공식을 적용:

```yaml
# configs/experiment/eomt/eomt-vitb.yaml
model:
  attn_mask_annealing_start_steps: [0, 0, 0, 0]
  attn_mask_annealing_end_steps: [200, 450, 700, 950]   # was [500, 1000, 1500, 2000]
```

이러면 deepest 블록(block 3)도 step ~950(epoch ~240)에 `prob=0`에 도달 → best 체크포인트(epoch 394 ≈ step 1580) 전에 약 150 epoch의 maskless 학습 여유가 생긴다. 그 시점엔 4개 블록 모두 prob=0이라 masked와 maskless가 동일 함수가 되어 gap이 사라진다.

> 주의: `S=1580`은 "옛 스케줄에서의 옛 best"다. end_steps를 바꾸면 학습 다이내믹스가 바뀌어 새 best 위치가 이동할 수 있다. 위 run에서 best가 step 1580으로 충분히 늦게 나왔으므로 새 best도 950보다 뒤일 가능성이 높지만, 보장은 아니다 → §6.1 재확인으로 확정한다.

### 5.2 (스킵) ViT-L `end_steps`

`eomt-vitl.yaml`은 여전히 `[40000, 80000, 120000, 160000]`이다. ViT-L은 best가 epoch 265(종료 285), 누적 step ~1,060 수준이라 4개 블록 모두 `prob≈1`(완전 masked)로 학습이 끝났을 가능성이 크다. **그러나 ViT-L은 FPS가 SegFormer 대비 5.36배 느려 Hailo 배포에서 제외**됐고, masked-eval 정확도 레퍼런스(mIoU 74.9%)로만 쓰인다. maskless로 export할 일이 없으므로 train/inference gap이 발생하지 않는다 → **지금 수정 불필요(스킵).** 단, 보고서에서 SegFormer와 공정 비교 시 그 74.9%가 masked-eval 수치임을 명시하는 것이 정직하다. 나중에 ViT-L을 다시 학습할 계획이 생기면 같은 공식으로 당기면 된다.

### 5.3 (선택) 검증을 배포 조건(maskless)으로 맞추기 — 보험

end_steps 캘리브레이션이 빗나가도 안전하게 만드는 보험. `validation_step`에서만 마스크를 끄면 early stopping과 체크포인트 선택이 **배포 동작 기준**으로 best를 고른다. ViT-B의 새 best가 step 950 이후에 나오면 5.1만으로 충분하므로 필수는 아니다.

`src/models/base.py` → `EoMTLightningModule.validation_step` 내부의 `self(x)` 호출부를 감싼다:
```python
model_ref = self.model.module if hasattr(self.model, "module") else self.model
prev = model_ref.masked_attn_enabled
model_ref.masked_attn_enabled = False        # 배포와 동일하게 maskless로 평가
try:
    mask_logits_list, class_logits_list = self(x)
finally:
    model_ref.masked_attn_enabled = prev      # 학습 step은 계속 masked 유지
```

---

## 6. 검증 방법

### 6.1 체크포인트의 annealing 상태 확인 (§4에서 사용한 스크립트)

`attn_mask_probs`는 버퍼로 등록되어 체크포인트 `state_dict`에 저장된다.
```python
import torch
ckpt = torch.load("<체크포인트 경로>", map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
for k, v in sd.items():
    if "attn_mask_probs" in k:
        print(k, v)
print("global_step:", ckpt.get("global_step"))
```
4개 값이 모두 ≈0이면 안전, `[~0, ~0, ~0, >0]`이면 deepest 블록이 덜 풀린 것.

### 6.2 "마스크 빼도 안전한가"를 직접 측정 (재학습 결정용)

같은 체크포인트로 `masked_attn_enabled=True` vs `False` 두 번 evaluate를 돌려 mIoU를 비교한다.
- 두 값이 거의 같으면 → maskless export 안전 (재학습 불필요).
- `False`가 크게 떨어지면 → annealing 부족. (이미 고친) `end_steps`로 재학습 후 §6.1로 재확인.

---

## 7. 결론 및 다음 단계

| 질문 | 답 |
|------|----|
| 코드에 버그가 있나? | 아니요 |
| 진단이 확정됐나? | 예 — block 3 = 0.2455 (예측 0.25와 일치) |
| ViT-B를 어떻게 고쳤나? | `end_steps` → `[200, 450, 700, 950]` (S=1580 기준) |
| ViT-L도 고쳐야 하나? | 아니요 — 배포 안 함(레퍼런스용). 재학습 계획 시에만 |
| 검증 토글(5.3)은 필수인가? | 아니요 — 보험. 새 best가 step 950 이후면 5.1만으로 충분 |
| 기존 ONNX(epoch 394)는? | block 3에 잔여 gap 있음 → §6.2로 손해 측정 후 재학습 여부 결정 |

**핵심:** config 수정은 **다음 학습 run부터** 효과가 있다. 현재 export된 ONNX는 여전히 block 3 gap을 가진 상태이므로, (1) §6.2로 기존 체크포인트의 maskless 손해를 먼저 측정 → 미미하면 그대로 사용, 유의미하면 (2) 고친 `end_steps`로 재학습 후 §6.1로 새 best의 `attn_mask_probs`가 4개 모두 ≈0인지 확인하고 export.
