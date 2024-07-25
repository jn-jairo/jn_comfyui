from comfy.sd import *
import comfy.sd

def patchComfySd(config):
    decode_tile_size_cache = {}
    encode_tile_size_cache = {}

    def get_decode_tile_size(idx, default):
        if idx not in decode_tile_size_cache:
            decode_tile_size_cache[idx] = default
        return decode_tile_size_cache[idx]

    def set_decode_tile_size(idx, tile_size):
        decode_tile_size_cache[idx] = tile_size

    def get_encode_tile_size(idx, default):
        if idx not in encode_tile_size_cache:
            encode_tile_size_cache[idx] = default
        return encode_tile_size_cache[idx]

    def set_encode_tile_size(idx, tile_size):
        encode_tile_size_cache[idx] = tile_size

    def decode(self, samples_in):
        pixel_samples = None

        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            model_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], self.output_channels) + tuple(map(lambda a: a * self.upscale_ratio, samples_in.shape[2:])), device=self.output_device)
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x:x+batch_number] = self.process_output(self.first_stage_model.decode(samples).to(self.output_device).float())
        except model_management.OOM_EXCEPTION as e:
            pixel_samples = None

            idx = "_".join([str(i) for i in samples_in.shape])

            orig_tile_size = tile_size = get_decode_tile_size(idx, 128 if len(samples_in.shape) == 3 else 64)

            while tile_size >= 8:
                model_management.soft_empty_cache(True)

                overlap = tile_size // 4
                logging.warning(f"Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding with tile size {tile_size} and overlap {overlap}.")
                try:
                    if len(samples_in.shape) == 3:
                        pixel_samples = self.decode_tiled_1d(samples_in, tile_x=tile_size, overlap=overlap)
                    else:
                        pixel_samples = self.decode_tiled_(samples_in, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
                    if tile_size < orig_tile_size:
                        set_decode_tile_size(idx, tile_size)
                    break
                except model_management.OOM_EXCEPTION:
                    pass
                tile_size -= 8

            if pixel_samples is None:
                raise e

        pixel_samples = pixel_samples.to(self.output_device).movedim(1,-1)
        return pixel_samples

    def encode(self, pixel_samples):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1,1)
        samples = None

        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            model_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)
            samples = torch.empty((pixel_samples.shape[0], self.latent_channels) + tuple(map(lambda a: a // self.downscale_ratio, pixel_samples.shape[2:])), device=self.output_device)
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = self.process_input(pixel_samples[x:x+batch_number]).to(self.vae_dtype).to(self.device)
                samples[x:x+batch_number] = self.first_stage_model.encode(pixels_in).to(self.output_device).float()

        except model_management.OOM_EXCEPTION as e:
            samples = None

            idx = "_".join([str(i) for i in pixel_samples.shape])

            orig_tile_size = tile_size = get_encode_tile_size(idx, 128 * 2048 if len(pixel_samples.shape) == 3 else 512)

            while tile_size >= 64:
                model_management.soft_empty_cache(True)

                overlap = tile_size // 8
                logging.warning(f"Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding with tile size {tile_size} and overlap {overlap}.")
                try:
                    if len(pixel_samples.shape) == 3:
                        samples = self.encode_tiled_1d(pixel_samples, tile_x=tile_size, overlap=overlap)
                    else:
                        samples = self.encode_tiled_(pixel_samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
                    if tile_size < orig_tile_size:
                        set_encode_tile_size(idx, tile_size)
                    break
                except model_management.OOM_EXCEPTION:
                    pass
                tile_size -= 64

            if samples is None:
                raise e

        return samples

    comfy.sd.VAE.decode = decode
    comfy.sd.VAE.encode = encode

PATCHES = {
    "60_comfy_sd": patchComfySd,
}
