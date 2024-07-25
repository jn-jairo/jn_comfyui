from comfy.ldm.modules.attention import *
import comfy.ldm.modules.attention

def patchComfyLdmModulesAttention(config):
    steps_cache = {}

    def get_steps(idx):
        if idx not in steps_cache:
            steps_cache[idx] = 1
        return steps_cache[idx]

    def set_steps(idx, steps):
        steps_cache[idx] = steps

    def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
        attn_precision = get_attn_precision(attn_precision)

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        scale = dim_head ** -0.5

        h = heads
        if skip_reshape:
            q, k, v = map(
                lambda t: t.reshape(b * heads, -1, dim_head),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, -1, heads, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * heads, -1, dim_head)
                .contiguous(),
                (q, k, v),
            )

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = model_management.get_free_memory(q.device)

        if attn_precision == torch.float32:
            element_size = 4
            upcast = True
        else:
            element_size = q.element_size()
            upcast = False

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
        modifier = 3
        if config["memory_estimation_multiplier"] is not None and config["memory_estimation_multiplier"] >= 0:
            modifier = config["memory_estimation_multiplier"]
        mem_required = tensor_size * modifier

        idx = f"{mem_required}_{q.shape[1]}"

        orig_steps = steps = get_steps(idx)

        if steps == 1 and mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        if steps >= q.shape[1]:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

        if mask is not None:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

        first_op_done = False
        cleared_cache = False
        while True:
            try:
                slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size
                    if upcast:
                        with torch.autocast(enabled=False, device_type = 'cuda'):
                            s1 = einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * scale
                    else:
                        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * scale

                    if mask is not None:
                        if len(mask.shape) == 2:
                            s1 += mask[i:end]
                        else:
                            s1 += mask[:, i:end]

                    s2 = s1.softmax(dim=-1).to(v.dtype)
                    del s1
                    first_op_done = True

                    r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                    del s2
                if steps > orig_steps:
                    set_steps(idx, steps)
                break
            except model_management.OOM_EXCEPTION as e:
                if first_op_done == False:
                    model_management.soft_empty_cache(True)
                    if cleared_cache == False:
                        cleared_cache = True
                        logging.warning("out of memory error, emptying cache and trying again")
                        continue

                    steps += 1
                    while (q.shape[1] % steps) != 0 and steps < q.shape[1]:
                        steps += 1
                    if steps >= q.shape[1]:
                        raise e

                    logging.warning("out of memory error, increasing steps and trying again {}".format(steps))
                else:
                    raise e

        del q, k, v

        r1 = (
            r1.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return r1

    if comfy.ldm.modules.attention.optimized_attention is comfy.ldm.modules.attention.attention_split:
        comfy.ldm.modules.attention.optimized_attention = attention_split

    if comfy.ldm.modules.attention.optimized_attention_masked is comfy.ldm.modules.attention.attention_split: 
        comfy.ldm.modules.attention.optimized_attention_masked = attention_split

    comfy.ldm.modules.attention.attention_split = attention_split

PATCHES = {
    "50_comfy_ldm_modules_attention": patchComfyLdmModulesAttention,
}
