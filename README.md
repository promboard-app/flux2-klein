# FLUX.2 [klein] 4B — 글로 만들기 · 참조 이미지로 고치기

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **FLUX.2 klein 생성 · 편집** 노드 하나를 더합니다. 프롬프트만 이으면 글로 이미지 한 장(PNG)을 만들고, 참조 이미지(4장까지)를 같이 이으면 그 이미지를 고치거나 여러 장을 섞습니다. 모델은 Black Forest Labs 의 [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)입니다.

> **검증 안 됨.** PromBoard 팀은 이 플러그인을 만들기만 하고 실제로 돌려 보지 않았습니다. 결과 그림이 없는 것도 그래서입니다. 코드는 diffusers 0.42.0 의 공식 파이프라인(Flux2KleinPipeline)과 GGUF 읽기 기능을 그대로 씁니다.

## 경량화 (판과 정밀도)

노드의 **경량화** 알약에서 고릅니다. 고른 판의 파일만 받습니다. **자동**은 이 PC 그래픽카드 메모리로 증류판을 고릅니다 — 12GB 이상 원본(bf16), 8GB 이상 GGUF Q8, 그 밖(맥 · AMD · 'CPU 만 쓰기' 포함)은 GGUF Q4.

| 경량화 알약 | 판 | 트랜스포머 | 그래픽카드 메모리 |
|---|---|---|---|
| 증류판 · 원본 (bf16) | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 7.8GB | 12GB 이상 |
| 증류판 · GGUF Q8 | 같은 판 | 4.3GB | 8GB 이상 |
| 증류판 · GGUF Q4 | 같은 판 | 2.6GB | 6GB 이상 |
| Base · 원본 (bf16) | [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | 7.8GB | 12GB 이상 |
| Base · GGUF Q8 | 같은 판 | 4.3GB | 8GB 이상 |
| Base · GGUF Q4 | 같은 판 | 2.6GB | 6GB 이상 |

- 증류판은 4단계 · guidance 1.0 으로 빠르고, Base 는 증류하지 않은 판이라 50단계 · guidance 4.0 으로 느린 대신 결과가 더 다양합니다(값은 공식 카드 예시).
- GGUF 판은 [unsloth](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) 가 줄인 파일입니다(Q8_0 · Q4_K_M). 원본보다 메모리를 덜 쓰는 대신 그림이 조금 거칠 수 있습니다.
- 글 인코더(Qwen3-4B, bf16 8.0GB) · 토크나이저 · VAE · 스케줄러는 공식 FLUX.2-klein-4B 저장소의 것을 모든 판이 같이 씁니다(두 공식 저장소의 이 파일들은 sha256 까지 같습니다).
- 그래픽카드 메모리 값은 트랜스포머 크기에 계산 몫을 더한 어림값입니다(실측 아님).

## 설정

| 알약 | 뜻 |
|---|---|
| 크기 | 자동(첫 참조 이미지 크기, 참조 이미지가 없으면 1024×1024) · 정사각 1024 · 가로/세로 16:9 · 4:3 · 3:2 · 정사각 1536(느림) |
| 시드 | -1 이면 매번 다른 그림. 쓴 시드는 PNG 정보(parameters)에 남습니다 |
| 단계 수 | 0 이면 판 기본값(증류판 4 · Base 50) |
| 프롬프트 따르기 (Base) | Base 만. 공식 예시 4. 증류판은 1 로 고정 |

참조 이미지는 파이프라인이 한 장씩 100만 픽셀 안으로 줄여 씁니다. 투명한 PNG 는 흰 바탕에 얹어 넘깁니다.

## 받는 것과 용량

| 고른 판 | 처음 받는 양 |
|---|---|
| 증류판 · 원본 (bf16) | 15.98GB(공통 8.23GB + 트랜스포머 7.75GB) |
| 증류판 · GGUF Q8 · Q4 | 12.53GB · 10.83GB |
| Base · 원본 · GGUF Q8 · Q4 | 15.98GB · 12.53GB · 10.83GB |

판을 바꾸면 그 판의 트랜스포머만 더 받습니다.

## 필요한 그래픽카드 · 메모리

- 플러그인은 부품을 그래픽카드 메모리에 맞게 올립니다. 다 들어가면 전부 올리고, 아니면 글 인코더 → 트랜스포머 → VAE 를 쓸 차례에만 올립니다. 혼자도 안 들어가는 부품은 조각씩 올립니다(느려집니다). GGUF 트랜스포머는 조각씩 올리지 않고 "더 가벼운 판으로 바꾸라"는 오류를 냅니다.
- 시스템 메모리(RAM)는 32GB 이상을 권합니다(가중치 최대 약 16GB 를 메모리에 올립니다).
- NVIDIA(CUDA) 그래픽카드용입니다. CPU 로는 너무 느려 막았습니다. 리눅스 · 맥(애플 칩) 잠금 파일도 있지만 돌려 보지 않았습니다.

## 안전 필터

공식 카드는 4B 를 쓰는 쪽에 NSFW 필터를 쓰라고 **권장**합니다(의무가 아닙니다. 9B 비상업판은 약관상 의무). 이 플러그인은 필터를 넣지 않았습니다. BFL 이 공개한 필터는 별도의 큰 비전 언어 모델을 하나 더 받아 돌려야 해서 이 노드의 메모리 · 받는 양을 크게 늘리기 때문입니다. 결과물을 공개 서비스에 쓰는 쪽은 자기 필터를 따로 두세요.

## 라이선스와 조건

- 모델: Apache-2.0(FLUX.2-klein-4B · FLUX.2-klein-base-4B · unsloth GGUF 모두). 상업 사용 · 지역 제한 조건이 없습니다.
- 이 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).

## 설치

1. PromBoard 앱(1.0.3 이상) → 플러그인 매니저 → **커뮤니티 플러그인**에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 실행 준비가 파이썬 패키지(diffusers 0.42.0 · transformers 5.20.2 · accelerate · gguf, torch 는 앱이 고정한 2.8.0)와 고른 판의 모델을 받습니다.
4. 캔버스 빈 곳을 우클릭하면 메뉴 맨 아래 플러그인 묶음에 **FLUX.2 klein 생성 · 편집**이 있습니다.

## English

A PromBoard community plugin that adds one node, **FLUX.2 klein 생성 · 편집** (generate). With a prompt only it renders one PNG; with up to four reference images it edits or combines them. Model: Black Forest Labs' [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

> **Unverified.** The PromBoard team built this plugin but did not run it, which is why there are no result pictures. The code uses diffusers 0.42.0's official Flux2KleinPipeline and its GGUF loader.

- **Quantization** setting picks the variant and precision; only the chosen files are downloaded. *Auto* picks the distilled model: bf16 on 12 GB+ VRAM, GGUF Q8 on 8 GB+, otherwise GGUF Q4 (also on Macs, AMD or CPU-only setups).
  - Distilled (4 steps, guidance 1.0) or Base (50 steps, guidance 4.0, more varied results), each as bf16 (7.8 GB transformer, ~12 GB VRAM), [unsloth](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) GGUF Q8 (4.3 GB, ~8 GB) or GGUF Q4 (2.6 GB, ~6 GB). VRAM figures are estimates, not measurements.
  - Text encoder (Qwen3-4B, 8.0 GB), tokenizer, VAE and scheduler come from the official FLUX.2-klein-4B repo and are shared by all options.
- Downloads: 15.98 GB (bf16) · 12.53 GB (Q8) · 10.83 GB (Q4). Switching options downloads only the new transformer.
- Memory: everything goes on the GPU when it fits; otherwise each part is moved on only while it runs and streamed block by block if needed (GGUF transformers are not streamed; the node asks for a lighter option instead). 32 GB+ system RAM recommended. NVIDIA CUDA; CPU is blocked; Linux and Apple-silicon lock files exist but are untested.
- Safety filter: the official card *recommends* (does not require) an NSFW filter for the 4B model. This plugin does not include one, because BFL's filter needs an extra large vision-language model. Add your own filter if you publish outputs in a public service.
- License: Apache-2.0 (models and plugin code); no commercial or regional restrictions.

## License

Plugin code: Apache-2.0. Model weights are downloaded from their original sources at first run under their own license (Apache-2.0).
