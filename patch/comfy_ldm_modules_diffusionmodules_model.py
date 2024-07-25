from comfy.ldm.modules.diffusionmodules.model import *
import comfy.ldm.modules.diffusionmodules.model

def patchComfyLdmModulesDiffusionmodulesModel(config):
    steps_cache = {}

    def get_steps(idx):
        if idx not in steps_cache:
            steps_cache[idx] = 1
        return steps_cache[idx]

    def set_steps(idx, steps):
        steps_cache[idx] = steps

    def slice_attention(q, k, v):
        r1 = torch.zeros_like(k, device=q.device)
        scale = (int(q.shape[-1])**(-0.5))

        mem_free_total = model_management.get_free_memory(q.device)

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        if config["memory_estimation_multiplier"] is not None and config["memory_estimation_multiplier"] >= 0:
            modifier = config["memory_estimation_multiplier"]
        mem_required = tensor_size * modifier

        idx = f"{mem_required}_{q.shape[1]}"

        orig_steps = steps = get_steps(idx)

        if steps == 1 and mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        first_op_done = False
        cleared_cache = False
        while True:
            try:
                slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size
                    s1 = torch.bmm(q[:, i:end], k) * scale

                    s2 = torch.nn.functional.softmax(s1, dim=2).permute(0,2,1)
                    del s1
                    first_op_done = True

                    r1[:, :, i:end] = torch.bmm(v, s2)
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

        return r1

    comfy.ldm.modules.diffusionmodules.model.slice_attention = slice_attention

PATCHES = {
    "50_comfy_ldm_modules_diffusionmodules_model": patchComfyLdmModulesDiffusionmodulesModel,
}
