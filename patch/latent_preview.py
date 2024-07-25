from latent_preview import *
import latent_preview

def patchLatentPreview(config):
    class TAESDPreviewerImpl(LatentPreviewer):
        def __init__(self, taesd, device):
            self.taesd = taesd
            self.device = device

        def decode_latent_to_preview(self, x0):
            x_sample = self.taesd.decode(x0[:1].to(self.device))[0].movedim(0, 2)
            return preview_to_image(x_sample)

    latent_preview.TAESDPreviewerImpl = TAESDPreviewerImpl

    def get_previewer(device, latent_format):
        previewer = None
        method = args.preview_method
        if config["preview_device"] is not None:
            device = torch.device(config["preview_device"])
        if method != LatentPreviewMethod.NoPreviews:
            # TODO previewer methods
            taesd_decoder_path = None
            if latent_format.taesd_decoder_name is not None:
                taesd_decoder_path = next(
                    (fn for fn in folder_paths.get_filename_list("vae_approx")
                        if fn.startswith(latent_format.taesd_decoder_name)),
                    ""
                )
                taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

            if method == LatentPreviewMethod.Auto:
                method = LatentPreviewMethod.Latent2RGB

            if method == LatentPreviewMethod.TAESD:
                if taesd_decoder_path:
                    taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                    previewer = TAESDPreviewerImpl(taesd, device)
                else:
                    logging.warning("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

            if previewer is None:
                if latent_format.latent_rgb_factors is not None:
                    previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
        return previewer

    latent_preview.get_previewer = get_previewer

    def prepare_callback(model, steps, x0_output_dict=None):
        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = get_previewer(model.load_device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0

            preview_bytes = None
            if previewer:
                try:
                    preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
                except model_management.OOM_EXCEPTION as e:
                    pass
            pbar.update_absolute(step + 1, total_steps, preview_bytes)
        return callback

    latent_preview.prepare_callback = prepare_callback

PATCHES = {
    "70_latent_preview": patchLatentPreview,
}
