"""promboard.flux2-klein — FLUX.2 [klein] 4B(Black Forest Labs) 글 → 이미지 · 참조 이미지 편집 노드 코드.

노드 하나(generate): 프롬프트 글 + 참조 이미지(선택, 4장까지)를 받아 이미지 한 장(PNG)을 낸다.
  참조 이미지가 없으면 글 → 이미지, 있으면 그 이미지를 바탕으로 고치거나 여러 장을 섞는다(같은 모델이 둘 다 한다 — 공식 카드).
경량화 설정(quantization) — 판과 정밀도를 한 칸으로 고른다(manifest requires.models 의 when 이 고른 것만 받는다. '자동'은
  서버가 이 PC 그래픽카드 메모리로 distilled-bf16 · distilled-q8 · distilled-q4 중 하나로 풀어 넘긴다).
  판: distilled = FLUX.2-klein-4B 증류판, 4단계 · guidance 1.0(공식 카드 예시) — 빠르다
      base      = FLUX.2-klein-base-4B 증류하지 않은 판, 50단계 · guidance 4.0(공식 카드 예시) — 느린 대신 결과가 더 다양하다
  정밀도: bf16 = 공식 트랜스포머(7.8GB) · q8 / q4 = unsloth 가 GGUF Q8_0(4.3GB) · Q4_K_M(2.6GB) 으로 줄인 판
  판과 정밀도를 두 칸으로 나누지 않은 까닭: requires.models 의 when 은 설정 하나만 보므로, 고른 트랜스포머 파일만 받으려면 한 칸이어야 한다.
  글 인코더(Qwen3-4B, bf16 8GB) · 토크나이저 · VAE · 스케줄러는 공식 black-forest-labs/FLUX.2-klein-4B 의 것을 모든 판이 같이 쓴다
  (두 공식 저장소의 이 파일들은 sha256 이 같다). 두 판의 diffusers 설정 차이는 model_index.json 의 is_distilled 하나뿐이라,
  증류판 폴더로 만들고 그 값만 바꿔 넘긴다.
참조 이미지: 파이프라인이 한 장씩 100만 픽셀(1024²) 안으로 줄이고 16 의 배수로 자른다. 크기 '자동'이면 결과 크기는 첫 참조 이미지
  크기, 참조 이미지가 없으면 1024².

실행: diffusers 0.42.0 Flux2KleinPipeline + Flux2Transformer2DModel(bf16 은 from_pretrained, GGUF 는 from_single_file +
  GGUFQuantizationConfig bf16 계산).
그래픽카드 메모리에 맞게 부품을 올리는 법은 gpu.py. 파이프라인은 사이드카가 떠 있는 동안 다시 쓰고 판이 바뀌면 새로 만든다.
NSFW 필터: 공식 카드는 4B 를 쓰는 쪽에 필터를 '권장'(9B 비상업판은 의무)한다. 이 플러그인은 넣지 않았다 — README 「안전 필터」.
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from . import gpu  # noqa: E402

REPOS = {"distilled": "black-forest-labs/FLUX.2-klein-4B", "base": "black-forest-labs/FLUX.2-klein-base-4B"}
MODELS = {
    # 설정 값: (판, GGUF 모델 이름 = <저장소>/<파일>, bf16 이면 None — 그 판 공식 저장소의 transformer/ 를 읽는다)
    "distilled-bf16": ("distilled", None),
    "distilled-q8": ("distilled", "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf"),
    "distilled-q4": ("distilled", "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf"),
    "base-bf16": ("base", None),
    "base-q8": ("base", "unsloth/FLUX.2-klein-base-4B-GGUF/flux-2-klein-base-4b-Q8_0.gguf"),
    "base-q4": ("base", "unsloth/FLUX.2-klein-base-4B-GGUF/flux-2-klein-base-4b-Q4_K_M.gguf"),
}
DEFAULT_STEPS = {"distilled": 4, "base": 50}
MAX_REFERENCES = 4
MAX_SEQUENCE_LENGTH = 512  # 파이프라인 기본값
# 부품이 계산할 때 더 쓰는 VRAM(1024² 기준, 넓이에 비례해 늘린다). 트랜스포머는 참조 이미지 한 장마다 REF_RESERVE_GIB 를 더한다.
RESERVE_GIB = {"text_encoder": 0.5, "transformer": 1.5, "vae": 2.5}
REF_RESERVE_GIB = 0.5
SEED_MAX = 2 ** 31 - 1

_STATE = {"key": None, "pipe": None}


def repo_dir(ctx, variant, probe="scheduler/scheduler_config.json"):
    """받은 공식 저장소 폴더(<모델 폴더>/_plugin-models/<저장소>@<커밋>) — 그 저장소에서 받은 파일 자리(probe)에서 두 단계 위."""
    return os.path.dirname(os.path.dirname(ctx.model_path(f"{REPOS[variant]}/{probe}")))


def reserve_for(width, height, refs):
    """부품 이름 → 계산 중에 더 쓰는 VRAM(바이트). 트랜스포머 · VAE 몫은 1024² 를 기준으로 넓이에 비례해 늘린다."""
    scale = max(1.0, (width * height) / (1024 * 1024))
    out = {name: int(gib * gpu.GIB * (1.0 if name == "text_encoder" else scale)) for name, gib in RESERVE_GIB.items()}
    out["transformer"] += int(REF_RESERVE_GIB * gpu.GIB * refs)
    return out


def load_pipeline(ctx, model_key):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig

    variant, gguf_name = MODELS[model_key]
    distilled_dir = repo_dir(ctx, "distilled")
    dtype = torch.bfloat16
    ctx.progress(0.03, "load-model")
    if gguf_name:
        transformer = Flux2Transformer2DModel.from_single_file(
            ctx.model_path(gguf_name),
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            dtype=dtype,
            config=distilled_dir,
            subfolder="transformer",
            local_files_only=True,
        )
    else:
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo_dir(ctx, variant, "transformer/config.json"),
            subfolder="transformer",
            dtype=dtype,
            local_files_only=True,
        )
    ctx.progress(0.10, "load-model")
    pipe = Flux2KleinPipeline.from_pretrained(
        distilled_dir,
        transformer=transformer,
        is_distilled=(variant == "distilled"),
        dtype=dtype,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_pipeline(ctx, model_key, reserve):
    key = (model_key, ctx.device)
    if _STATE["key"] != key:
        drop_pipeline()
        pipe = load_pipeline(ctx, model_key)
        ctx.progress(0.16, "load-model")
        # GGUF 트랜스포머는 조각씩 올리기를 하지 않는다(gpu.py). bf16 트랜스포머는 혼자 안 들어가면 조각씩 올린다.
        no_group = ("transformer",) if MODELS[model_key][1] else ()
        plan = gpu.place(pipe, ["text_encoder", "transformer", "vae"], ctx.device, reserve, no_group=no_group)
        ctx.log(f"placement {plan}")
        _STATE.update(key=key, pipe=pipe)
    return _STATE["pipe"]


def drop_pipeline():
    old = _STATE["pipe"]
    _STATE.update(key=None, pipe=None)
    if old is not None:
        gpu.release(old)


def load_references(ctx, paths):
    from PIL import Image, ImageOps

    images = []
    for p in paths:
        with Image.open(ctx.in_path(p)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                # 투명한 곳은 흰 바탕으로 — 파이프라인은 RGB 만 받는다.
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                images.append(bg)
            else:
                images.append(im.convert("RGB"))
    for im in images:
        w, h = im.size
        if min(w, h) < 64 or max(w, h) > 8 * min(w, h):
            raise ValueError("참조 이미지는 짧은 변이 64픽셀 이상이고 가로세로 비가 8:1 안이어야 합니다.")
    return images


def oom_message(model_key):
    light = "" if model_key.endswith("-q4") else " 경량화를 더 가벼운 판(GGUF Q4)으로 바꾸거나"
    return (
        "그래픽카드 메모리(VRAM)가 모자라 이미지를 만들지 못했습니다." + light + " 크기를 줄이거나 참조 이미지 수를 줄이고,"
        " 그래픽카드를 쓰는 다른 프로그램(게임 · 다른 AI)을 닫은 뒤 다시 실행해 주세요."
    )


def generate(ctx, inputs, settings):
    import torch

    prompt = (inputs.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("프롬프트를 이어 주세요(프롬프트 노드 → 프롬프트 칸).")
    if ctx.device == "cpu":
        raise RuntimeError("FLUX.2 [klein] 은 그래픽카드(NVIDIA CUDA 또는 Apple Silicon)가 있어야 돕니다. CPU 로는 너무 느려 막아 두었습니다.")
    refs = inputs.get("images") or []
    if len(refs) > MAX_REFERENCES:
        raise ValueError(f"참조 이미지는 {MAX_REFERENCES}장까지 이어 주세요(지금 {len(refs)}장).")

    model_key = settings["quantization"]
    variant = MODELS[model_key][0]
    steps = int(settings["steps"]) or DEFAULT_STEPS[variant]
    guidance = float(settings["guidance"]) if variant == "base" else 1.0
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, SEED_MAX)
    size = settings["size"]
    width = height = None
    if size != "auto":
        width, height = (int(v) for v in size.split("x"))
    elif not refs:
        width = height = 1024

    images = load_references(ctx, refs) if refs else None
    reserve = reserve_for(width or 1024, height or 1024, len(refs))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        pipe = get_pipeline(ctx, model_key, reserve)

        def on_step(_pipe, i, _t, kwargs):
            ctx.check_cancel()
            ctx.progress(0.2 + 0.75 * (i + 1) / steps, "denoise")
            return kwargs

        ctx.progress(0.18, "encode")
        image = pipe(
            image=images,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            callback_on_step_end=on_step,
        ).images[0]
    except MemoryError as exc:
        ctx.log(f"placement failed: {exc}", "warn")
        drop_pipeline()
        raise RuntimeError(oom_message(model_key)) from None
    except torch.OutOfMemoryError:
        drop_pipeline()
        raise RuntimeError(oom_message(model_key)) from None
    except RuntimeError as exc:
        if not gpu.is_cpu_oom(exc):
            raise
        drop_pipeline()
        raise RuntimeError(gpu.RAM_MESSAGE) from None

    ctx.progress(0.98, "save")
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text(
        "parameters",
        f"{prompt}\nFLUX.2 klein 4B {model_key} · steps {steps} · guidance {guidance} · seed {seed} · {image.width}x{image.height} · references {len(refs)}",
    )
    out = ctx.out_path(f"flux2-klein-{seed}.png")
    image.save(out, pnginfo=info)
    report = gpu.peak_report()
    if report:
        ctx.log(report)
    ctx.progress(1.0, "done")
    return {"image": out}


NODES = {"generate": generate}
