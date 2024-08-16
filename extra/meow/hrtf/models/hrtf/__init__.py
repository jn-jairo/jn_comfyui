import sox
import numpy as np
import torch
import math

from ....base_model import BaseModel
from .hrtf import HRTF
from ....utils.audio_format import float2pcm, pcm2float

DEFAULT_CROSSFADE_MS = 25 # 40 times per s
DEFAULT_FREQUENCY_PER_S = 20 # every 50 ms
DEFAULT_DURATION_S = 1
DEFAULT_LOW_FREQ = 75

DEFAULT_AZIMUTH = 0
DEFAULT_ELEVATION = 0
DEFAULT_PROXIMITY = 1

class HrtfCipicModel(BaseModel):
    """
    Position:
        [azimuth, elevation]
        [  0,   0] corresponds to a point directly ahead
        [  0,  90] corresponds to a point directly overhead
        [  0, 180] corresponds to a point directly behind
        [  0, 270] corresponds to a point directly below
        [ 90,   0] corresponds to a point directly to the left
        [-90,   0] corresponds to a point directly to the right

        [proximity]
        [1.0] don't change
        [0.0] so far away you don't hear anything
        [10.0] so close and loud you'll go deaf
    """

    def load_model(self, *args, **kwargs):
        hrtf = HRTF()
        hrtf.load_subject(self.get_file_path())

        self.model = hrtf

        del hrtf

    def execute(self, audio=[], *args, **kwargs):
        if not isinstance(audio, list):
            audio = [audio]

        audios = []

        for a in audio:
            for x in self._execute(audio=a, *args, **kwargs):
                audios.append(x)

        return audios

    def _execute(self, audio, positions, frequency_per_s=DEFAULT_FREQUENCY_PER_S, crossfade_ms=DEFAULT_CROSSFADE_MS, low_freq=DEFAULT_LOW_FREQ, *args, **kwargs):
        """
        Position:
            {
                "azimuth": -90 to 90,
                "elevation": 0 to 270,
                "proximity": >= 0, = 1 dont change, < 1 further away, > 1 closer,
                "delay": >= 0 seconds,
            }
        """

        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"].clone().cpu()

        (batches, channels, samples) = waveform.shape

        transformer = sox.Transformer()
        transformer.remix(remix_dictionary={1: [x for x in range(1, channels + 1)]}, num_output_channels=1)
        transformer.set_output_format(channels=1)

        audios = []

        positions, duration_s = self.parse_positions(positions, frequency_per_s=frequency_per_s, crossfade_ms=crossfade_ms)

        angles = [[p["azimuth"], p["elevation"]] for p in positions]
        volume_gains = [p["proximity"] for p in positions]

        hrir_l, hrir_r = self.model.interpolater(angles)

        for b in range(batches):
            np_audio = waveform[b, :, :].clone().reshape(channels, samples).numpy()

            # to mono

            np_audio = transformer.build_array(input_array=np_audio.transpose(1, 0), sample_rate_in=audio["sample_rate"])

            if np_audio.ndim == 2:
                np_audio = np.squeeze(np_audio, axis=0)

            # to 3d stereo

            np_audio = float2pcm(np_audio)
            np_audio = self.model.multiple_convolve(
                samples=np_audio,
                hrir_l=hrir_l,
                hrir_r=hrir_r,
                sample_rate=sample_rate,
                circle_period=duration_s,
                crossfade_ms=crossfade_ms,
                low_freq=low_freq,
                volume_gains=volume_gains,
            )

            np_audio = pcm2float(np_audio.transpose(1, 0))

            audios.append({
                "sample_rate": sample_rate,
                "waveform": torch.from_numpy(np_audio.copy()).to(audio["waveform"].device).to(audio["waveform"].dtype).unsqueeze(0),
            })

            del np_audio

        del transformer, hrir_l, hrir_r

        return audios

    def parse_positions(self, positions, frequency_per_s=DEFAULT_FREQUENCY_PER_S, crossfade_ms=DEFAULT_CROSSFADE_MS):
        all_positions = []

        previous_position = None

        if positions is None:
            positions = []

        if len(positions) == 0:
            positions.append({
                "azimuth": 0,
                "elevation": 0,
                "proximity": 1,
                "delay": 1,
            })

        if len(positions) == 1:
            positions.append(positions[0])

        for position in positions:
            duration_s = position.get("delay", DEFAULT_DURATION_S)

            if duration_s == 0:
                ps = [{
                    "azimuth": position.get("azimuth", DEFAULT_AZIMUTH),
                    "elevation": position.get("elevation", DEFAULT_ELEVATION),
                    "proximity": position.get("proximity", DEFAULT_PROXIMITY),
                }]
            elif previous_position is None:
                previous_position = position
                continue
            else:
                ps = self.get_positions(
                    azimuth=previous_position.get("azimuth", DEFAULT_AZIMUTH),
                    azimuth_to=position.get("azimuth", DEFAULT_AZIMUTH),
                    elevation=previous_position.get("elevation", DEFAULT_ELEVATION),
                    elevation_to=position.get("elevation", DEFAULT_ELEVATION),
                    proximity=previous_position.get("proximity", DEFAULT_PROXIMITY),
                    proximity_to=position.get("proximity", DEFAULT_PROXIMITY),
                    duration_s=duration_s,
                    frequency_per_s=frequency_per_s,
                    crossfade_ms=crossfade_ms,
                )

            for p in ps:
                all_positions.append(p)

            del ps

            previous_position = position

        del previous_position

        frequency = 1 / frequency_per_s
        frequency = max(frequency, crossfade_ms / 1000)

        total_duration = len(all_positions) * frequency

        return all_positions, total_duration

    def get_positions(self, azimuth=0, azimuth_to=None, elevation=0, elevation_to=None, proximity=1, proximity_to=None, duration_s=DEFAULT_DURATION_S, frequency_per_s=DEFAULT_FREQUENCY_PER_S, crossfade_ms=DEFAULT_CROSSFADE_MS, action="move"):
        if azimuth_to is None:
            azimuth_to = azimuth

        if elevation_to is None:
            elevation_to = elevation

        if proximity_to is None:
            proximity_to = proximity

        azimuth = max(-90, min(90, azimuth))
        azimuth_to = max(-90, min(90, azimuth_to))

        elevation = max(0, min(270, elevation))
        elevation_to = max(0, min(270, elevation_to))

        proximity = max(0, proximity)
        proximity_to = max(0, proximity_to)

        frequency = 1 / frequency_per_s

        frequency = max(frequency, crossfade_ms / 1000)

        if action == "cycle":
            number_per_side = math.floor(duration_s/(2*frequency))
            number_samples = number_per_side * 2
        else:
            number_samples = int(duration_s/frequency)

        if duration_s/number_samples < crossfade_ms/1000:
            raise ValueError("Too fast to crossfade.")

        positions = []

        if action == "cycle":
            azimuths = np.linspace(azimuth, azimuth_to, number_per_side)
            azimuths_reverse = np.linspace(azimuth_to, azimuth, number_per_side)
            elevations = np.linspace(elevation, elevation_to, number_per_side)
            elevations_reverse = np.linspace(elevation_to, elevation, number_per_side)
            proximities = np.linspace(proximity, proximity_to, number_per_side)
            proximities_reverse = np.linspace(proximity_to, proximity, number_per_side)

            for i in range(number_per_side):
                positions.append({
                    "azimuth": azimuths[i],
                    "elevation": elevations[i],
                    "proximity": proximities[i],
                })

            for i in range(number_per_side):
                positions.append({
                    "azimuth": azimuths_reverse[i],
                    "elevation": elevations_reverse[i],
                    "proximity": proximities_reverse[i],
                })

        else:
            azimuths = np.linspace(azimuth, azimuth_to, number_samples)
            elevations = np.linspace(elevation, elevation_to, number_samples)
            proximities = np.linspace(proximity, proximity_to, number_samples)

            for i in range(number_samples):
                positions.append({
                    "azimuth": azimuths[i],
                    "elevation": elevations[i],
                    "proximity": proximities[i],
                })

        return positions

