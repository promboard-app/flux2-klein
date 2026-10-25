"""그래픽카드 메모리(VRAM)에 맞게 diffusers 파이프라인 부품을 올리는 도우미.

부품(글 인코더 · 트랜스포머 · VAE)을 실행 순서대로 본다. 부품마다 '계산 중에 더 쓰는 몫'(reserve)을 따로 받는다.
  - 다 합쳐(+ 가장 큰 몫) 남은 VRAM 에 들어가면 전부 그래픽카드에 둔다(가장 빠르다).
  - 아니면 부품마다 따로 정한다. 자기 몫까지 혼자 들어가는 부품은 '쓸 때만 올리기'(accelerate cpu_offload_with_hook — 쓸
    차례에 통째로 올리고 다음 부품이 쓰일 때 내린다, diffusers enable_model_cpu_offload 와 같은 방식). 혼자도 안 들어가는 부품은
    '조각씩 올리기'(diffusers apply_group_offloading block_level — 블록 하나씩 올려 계산한다. 느리지만 VRAM 이 적게 든다).
  - GGUF 로 줄인 트랜스포머는 조각씩 올리기를 하지 않는다(no_group). 혼자 안 들어가면 MemoryError — 노드가 더 가벼운 판을
    고르라는 오류 글로 바꾼다.
diffusers 는 한 파이프라인에 '쓸 때만 올리기'와 '조각씩 올리기'를 같이 걸지 못하게 막아서(enable_model_cpu_offload 가 거절)
같은 방식을 부품 단위로 직접 건다. 파이프라인 _all_hooks 는 비워 두어 maybe_free_model_hooks 가 손대지 않게 한다.
"""
import gc

GIB = 1024 ** 3


def module_bytes(module):
    """부품의 매개변수 + 버퍼 크기(바이트). GGUF 매개변수는 줄인 크기 그대로 센다."""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


def block_modules(module):
    """조각씩 올리기가 블록(ModuleList)을 찾아 들어갈 하위 모듈 이름들. diffusers block_level 은 바로 아래 자식만 보므로
    transformers 모델처럼 블록이 model.layers · model.language_model.layers 깊이에 있으면 그 길의 이름을 알려 줘야 한다."""
    import torch

    lists = (torch.nn.ModuleList, torch.nn.Sequential)
    names = set()

    def walk(m):
        for name, child in m.named_children():
            if isinstance(child, lists):
                continue
            if any(isinstance(x, lists) for x in child.modules()):
                names.add(name)
                walk(child)

    walk(module)
    return sorted(names)


def free_vram():
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free


def place(pipe, order, device, reserve, no_group=()):
    """파이프라인 부품을 device 에 맞게 올린다. 돌려주는 값: 부품 이름 → 'gpu' · 'hook' · 'group' · device.

    order     실행 순서의 부품 이름(예: ["text_encoder", "transformer", "vae"])
    reserve   부품 이름 → 그 부품이 계산할 때 더 쓰는 VRAM(활성값 · VAE 디코딩, 바이트)
    no_group  조각씩 올리기를 하지 않을 부품(GGUF 트랜스포머) — 혼자 안 들어가면 MemoryError
    """
    import torch

    if device != "cuda":
        pipe.to(device)
        return {name: device for name in order}

    from accelerate import cpu_offload_with_hook
    from diffusers.hooks import apply_group_offloading

    modules = {name: getattr(pipe, name) for name in order if getattr(pipe, name, None) is not None}
    sizes = {name: module_bytes(m) for name, m in modules.items()}
    free = free_vram()
    if sum(sizes.values()) + max(reserve.get(n, 0) for n in modules) <= free:
        for m in modules.values():
            m.to("cuda")
        return {name: "gpu" for name in modules}

    plan = {}
    hook = None
    cuda = torch.device("cuda")
    for name, m in modules.items():
        need = sizes[name] + reserve.get(name, 0)
        if need <= free:
            _, hook = cpu_offload_with_hook(m, cuda, prev_module_hook=hook)
            plan[name] = "hook"
        elif name in no_group:
            raise MemoryError(f"{name} needs {need / GIB:.1f} GiB, free {free / GIB:.1f} GiB")
        else:
            apply_group_offloading(
                m,
                onload_device=cuda,
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=True,
                low_cpu_mem_usage=True,
                block_modules=block_modules(m) or None,
            )
            plan[name] = "group"
    return plan


def release(pipe):
    """파이프라인을 내려놓고 VRAM 을 비운다(모델을 바꿀 때 · 메모리가 모자랄 때). 부르는 쪽은 자기 참조를 먼저 지운다."""
    import torch

    try:
        from accelerate.hooks import remove_hook_from_module

        for m in pipe.components.values():
            if isinstance(m, torch.nn.Module):
                remove_hook_from_module(m, recurse=True)
                m.to("cpu")
    except Exception:
        pass
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cpu_oom(exc):
    """컴퓨터 메모리(RAM) 부족 — torch CPU 할당 실패(DefaultCPUAllocator)나 파이썬 MemoryError 가 아닌 RuntimeError 꼴."""
    text = str(exc)
    return "DefaultCPUAllocator" in text or "not enough memory" in text


RAM_MESSAGE = (
    "컴퓨터 메모리(RAM)가 모자라 모델을 옮기지 못했습니다. 메모리를 많이 쓰는 다른 프로그램을 닫고 다시 실행해 주세요."
    " 모델 파일을 모두 메모리에 올리므로 RAM 이 모델 크기보다 넉넉해야 합니다(README 「필요한 메모리」)."
)


def peak_report():
    """이번 실행의 VRAM 최대(할당 · 예약, MiB) — 로그 한 줄용."""
    import torch

    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.max_memory_allocated() // (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() // (1024 * 1024)
    return f"VRAM peak allocated {alloc} MiB · reserved {reserved} MiB"
