from scipy.io import loadmat
from scipy.spatial import Delaunay
from scipy.signal import butter, lfilter, convolve
import numpy as np

class HRTF(object):
    AZIMUTHES = 25
    ELEVATIONS = 50
    INSTANTS_IN_TIME = 200

    def __init__(self):
        self.hrir = {}
        self.triangulation = {'points': [],
                              'triangles' : None}
    def weight_calc(self,points):
        tri = self.triangulation['triangles'].find_simplex(points)
        X = self.triangulation['triangles'].transform[tri,:2]
        Y = points - self.triangulation['triangles'].transform[tri,2]
        b = np.einsum('ijk,ik->ij', X, Y)
        return (np.c_[b,1-b.sum(axis=1)],
                self.triangulation['triangles'].simplices[tri])

    def load_subject(self,subject_file,hrir_len=200,azimuths=None,elevations=None):
        x = loadmat(subject_file)
        hrir_r = x['hrir_r']
        hrir_l = x['hrir_l']
        ir = {'L': {}, 'R': {}}
        if azimuths is None:
            azimuths = [-80, -65, -55, -45, -40, -35, -30, -25, -20,
					-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80]
        if elevations is None:
            elevations = [-45+5.625*x for x in range(50)]
        points = []

        for azi in [-90,90]:
            ir['L'][azi] = {}
            ir['R'][azi] = {}
            if azi == -90:
                for j,elv in enumerate(elevations):
                    if j == 0:
                        to_add_l = hrir_l[0,j]
                        to_add_r = hrir_r[0,j]
                    else:
                        to_add_l += hrir_l[0,j]
                        to_add_r += hrir_r[0,j]
                avg_l = to_add_l/50
                avg_r = to_add_r/50
                all_elvs = [x for x in elevations]
                all_elvs.extend([-90,270])
                for elv in all_elvs:
                    ir['L'][azi][elv] = avg_l
                    ir['R'][azi][elv] = avg_r
                    points.append([azi,elv])
            else:
                for j,elv in enumerate(elevations):
                    if j == 0:
                        to_add_l = hrir_l[-1,j]
                        to_add_r = hrir_r[-1,j]
                    else:
                        to_add_l += hrir_l[-1,j]
                        to_add_r += hrir_r[-1,j]
                avg_l = to_add_l/50
                avg_r = to_add_r/50
                for elv in all_elvs:
                    ir['L'][azi][elv] = avg_l
                    ir['R'][azi][elv] = avg_r
                    points.append([azi,elv])

        for i,azi in enumerate(azimuths):
            ir['L'][azi] = {}
            ir['R'][azi] = {}

            ir['L'][azi][-90] = self.calculate_weighted(hrir_l[i,0],-45.0,
                                                hrir_l[i,-1],230.625)
            ir['R'][azi][-90] = self.calculate_weighted(hrir_r[i,0],-45.0,
                                                hrir_r[i,-1],230.625)
            points.append([azi,-90])
            for j,elv in enumerate(elevations):
                ir['L'][azi][elv] = hrir_l[i,j]
                ir['R'][azi][elv] = hrir_r[i,j]
                points.append([azi,elv])
            ir['L'][azi][270] = ir['L'][azi][-90]
            ir['R'][azi][270] = ir['R'][azi][-90]
            points.append([azi,270])

        self.hrir = ir
        self.triangulation['triangles'] = Delaunay(np.array(points))
        self.triangulation['points'] = points

    def interpolater(self, angles):
        weights,indices = self.weight_calc(angles)
        points = []
        for triangle in indices:
            triangle_points = []
            for index in triangle:
                triangle_points.append(self.triangulation['points'][index])
            points.append(triangle_points)
        right = self.hrir['R']
        left = self.hrir['L']
        interp_R = []
        interp_L = []
        def interpolate_loop(right,left,weights,points,interp_R,interp_L):
            for i in range(weights.shape[0]):
                interp_R.append(weights[i][0]*right[points[i][0][0]][points[i][0][1]]+
                                weights[i][1]*right[points[i][1][0]][points[i][1][1]]+
                                weights[i][2]*right[points[i][2][0]][points[i][2][1]])

                interp_L.append(weights[i][0]*left[points[i][0][0]][points[i][0][1]]+
                                weights[i][1]*left[points[i][1][0]][points[i][1][1]]+
                                weights[i][2]*left[points[i][2][0]][points[i][2][1]])
        interpolate_loop(right,left,weights,points,interp_R,interp_L)
        return (interp_R,interp_L)

    def multiple_convolve(self, samples, hrir_l, hrir_r, sample_rate, circle_period, crossfade_ms=25, low_freq=50, volume_gains=None):
        crossfade_amount = int(sample_rate*crossfade_ms/1000.)
        left_channel = []
        right_channel = []
        tailed_left = []
        tailed_right = []
        total_angles = len(hrir_l)

        if volume_gains is None:
            volume_gains = [1.0] * total_angles

        i = 0
        pos = 0
        angle_time = int(circle_period/total_angles*sample_rate)
        total_samples = len(samples)
        left = self.butter_highpass_filter(samples,low_freq,sample_rate)
        right = self.butter_highpass_filter(samples,low_freq,sample_rate)
        save = self.butter_lowpass_filter(samples,low_freq,sample_rate)

        convolution_margin = self.INSTANTS_IN_TIME - 1

        while pos < total_samples:
            end = pos + angle_time + convolution_margin
            gain = volume_gains[i]

            # apply convolution and multiply by volume gain
            left_conv = convolve(left[pos:end],hrir_l[i],'valid')
            right_conv = convolve(right[pos:end],hrir_r[i],'valid')

            left_conv = self.apply_gain(gain, left_conv, sample_rate)
            right_conv = self.apply_gain(gain, right_conv, sample_rate)

            # ensure it does not exceed the int16 range after applying the gain
            # left_conv = np.clip(left_conv, -32768, 32767)
            # right_conv = np.clip(right_conv, -32768, 32767)

            left_channel.append(left_conv)
            right_channel.append(right_conv)

            # apply convolution and multiply by volume gain
            tailed_left_conv = convolve(left[pos:end+crossfade_amount], hrir_l[i],'valid')
            tailed_right_conv = convolve(right[pos:end+crossfade_amount], hrir_r[i],'valid')

            tailed_left_conv = self.apply_gain(gain, tailed_left_conv, sample_rate)
            tailed_right_conv = self.apply_gain(gain, tailed_right_conv, sample_rate)

            # ensure it does not exceed the int16 range after applying the gain
            # tailed_left_conv = np.clip(tailed_left_conv, -32768, 32767)
            # tailed_right_conv = np.clip(tailed_right_conv, -32768, 32767)

            tailed_left.append(tailed_left_conv)
            tailed_right.append(tailed_right_conv)

            pos += angle_time
            i += 1
            i %= total_angles

            del left_conv, right_conv, tailed_left_conv, tailed_right_conv

        left_channel,right_channel = self.crossfade_tails(left_channel,
                                                    right_channel,tailed_left,
                                                    tailed_right,crossfade_amount)
        left_channel = np.array(left_channel)
        right_channel = np.array(right_channel)
        # left_channel += save[:-convolution_margin]
        # right_channel += save[:-convolution_margin]
        left_channel += save[:left_channel.shape[0]]
        right_channel += save[:right_channel.shape[0]]

        # the convolution reduces the amount of samples, so let's complete with zeros

        output_length = len(left_channel)
        padding_length = total_samples - output_length

        if padding_length > 0:
            zero_padding = np.zeros(padding_length)
            left_channel = np.concatenate((left_channel, zero_padding))
            right_channel = np.concatenate((right_channel, zero_padding))

        # ensure it does not exceed the int16 range after adding the saved samples

        left_channel = np.clip(left_channel, -32768, 32767).astype(np.int16)
        right_channel = np.clip(right_channel, -32768, 32767).astype(np.int16)

        return np.column_stack((left_channel,right_channel))

    def apply_gain(self, gain, samples, sample_rate):
        if gain < 0:
            # reduce more the high frequencies to sound like it's far away

            original_gain = gain
            cutoff_freq = 1000
            decay_multiplier = 20
            slope_db_per_octave = 6
            slope_db_per_octave += (1 - original_gain) * decay_multiplier

            # Convert audio to frequency domain
            n = len(samples)
            freq_data = fft(samples)
            freqs = np.fft.fftfreq(n, d=1/sample_rate)

            # Create a frequency-dependent gain
            gain = np.ones_like(freqs)
            above_cutoff = np.abs(freqs) > cutoff_freq
            gain[above_cutoff] = 10**(-slope_db_per_octave * np.log2(np.abs(freqs[above_cutoff]) / cutoff_freq) / 20)

            # apply the gain to the slope
            gain = gain * original_gain

            # Apply gain to frequency domain data
            freq_data_adjusted = freq_data * gain

            # Convert back to time domain
            samples = np.real(ifft(freq_data_adjusted))

        else:
            samples = samples * gain

        return samples

    def crossfade_tails(self, left,right,tailed_left,tailed_right,filter_len):
        final_left = []
        final_right = []
        t = np.linspace(0,np.pi/2,filter_len)
        fade_out = np.cos(t)**2
        fade_in = np.sin(t)**2
        for i, left_block in enumerate(left):
            if i == 0:
                final_left.extend(left_block)
                final_right.extend(right[i])
            else:
                if len(left_block) < filter_len:
                    t = np.linspace(0,np.pi/2,len(left_block))
                    filter_len = len(left_block)
                    fade_out = np.cos(t)**2
                    fade_in = np.sin(t)**2
                faded_left = tailed_left[i-1][-filter_len:]*fade_out + left_block[:filter_len]*fade_in
                faded_right = tailed_right[i-1][-filter_len:]*fade_out + right[i][:filter_len]*fade_in
                left_block[:filter_len] = faded_left
                right[i][:filter_len] = faded_right
                final_left.extend(left_block)
                final_right.extend(right[i])
        return final_left, final_right

    def calculate_weighted(self, vect1,point1,vect2,point2):
        total = abs(-90 - point1) + abs(270 - point2)
        weight1 = abs(-90 - point1)/total
        weight2 = abs(270 - point2)/total
        vect1 *= weight1
        vect2 *= weight2
        return vect1+vect2

    def butter_pass(self, cutoff,sr,filt_type,order=5):
        nyq = 0.5*sr
        cutoff = cutoff / nyq
        b, a = butter(order,cutoff, btype=filt_type, analog=False)
        return b,a

    def butter_lowpass_filter(self, data,lowcutoff,sr,order=5):
        b,a = self.butter_pass(lowcutoff,sr,'low',order=order)
        y = lfilter(b,a,data)
        return y

    def butter_highpass_filter(self, data,highcutoff,sr,order=5):
        b,a = self.butter_pass(highcutoff,sr,'high',order)
        y = lfilter(b,a,data)
        return y

