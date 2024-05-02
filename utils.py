# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Some Useful Common Methods

import os
from pathlib import Path
import pandas as pd
import scipy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import logging
import os
import sys
import h5py
import csv
import time
import json
# import museval
import librosa
from datetime import datetime
from tqdm import tqdm
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Uniform, Beta
from torchaudio import transforms as T
import random
import torchaudio
from typing import Union, Tuple, Optional

import sed_eval
import psds_eval
from psds_eval import PSDSEval, plot_psd_roc
import sed_scores_eval
from sed_scores_eval.utils.scores import create_score_dataframe

# import from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x # without sigmoid since it has been computed
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)]
    return np.array(mixup_lambdas).astype(np.float32)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def dump_config(config, filename, include_time = False):
    save_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_json = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_json[key] = eval("config." + key)
    if include_time:
        filename = filename + "_" + save_time
    with open(filename + ".json", "w") as f:      
        json.dump(config_json, f ,indent=4)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min = -1., a_max = 1.)
    return (x * 32767.).astype(np.int16)


# index for each class
def process_idc(index_path, classes_num, filename):
    # load data
    logging.info("Load Data...............")
    idc = [[] for _ in range(classes_num)]
    with h5py.File(index_path, "r") as f:
        for i in tqdm(range(len(f["target"]))):
            t_class = np.where(f["target"][i])[0]
            for t in t_class:
                idc[t].append(i)
    print(idc)
    np.save(filename, idc)
    logging.info("Load Data Succeed...............")

def clip_bce(pred, target):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(pred, target)
    # return F.binary_cross_entropy(pred, target)


def clip_ce(pred, target):
    return F.cross_entropy(pred, target)

def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


def get_loss_func(loss_type):
    if (loss_type == 'clip_bce') or (loss_type == 'frame_bce'):
        return clip_bce
    if loss_type == 'clip_ce':
        return clip_ce
    if loss_type == 'asl_loss':
        loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=0,clip=0.05)
        return loss_func

def do_mixup_label(x):
    out = torch.logical_or(x, torch.flip(x, dims = [0])).float()
    return out

def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0,-1) * mixup_lambda + torch.flip(x, dims = [0]).transpose(0,-1) * (1 - mixup_lambda)).transpose(0,-1)
    return out
    
def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

# set the audio into the format that can be fed into the model
# resample -> convert to mono -> output the audio  
# track [n_sample, n_channel]
def prepprocess_audio(track, ofs, rfs, mono_type = "mix"):
    if track.shape[-1] > 1:
        # stereo
        if mono_type == "mix":
            track = np.transpose(track, (1,0))
            track = librosa.to_mono(track)
        elif mono_type == "left":
            track = track[:, 0]
        elif mono_type == "right":
            track = track[:, 1]
    else:
        track = track[:, 0]
    # track [n_sample]
    if ofs != rfs:
        track = librosa.resample(track, ofs, rfs)
    return track

def init_hier_head(class_map, num_class):
    class_map = np.load(class_map, allow_pickle = True)
    
    head_weight = torch.zeros(num_class,num_class).float()
    head_bias = torch.zeros(num_class).float()

    for i in range(len(class_map)):
        for d in class_map[i][1]:
            head_weight[d][i] = 1.0
        for d in class_map[i][2]:
            head_weight[d][i] = 1.0 / len(class_map[i][2])
        head_weight[i][i] = 1.0
    return head_weight, head_bias

def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)

        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture

def pad_audio(audio, target_len, fs):
    if audio.shape[-1] < target_len:
        rand_onset = random.randint(0, target_len - audio.shape[-1])
        padded_audio = torch.zeros(target_len, dtype = audio.dtype)
        padded_audio[rand_onset:rand_onset + audio.shape[-1]] = audio
        
        onset_s = round(rand_onset / fs, 3)
        offset_s = round((rand_onset + audio.shape[-1]) / fs, 3)
    
    elif len(audio) > target_len:
        rand_onset = random.randint(0, len(audio) - target_len)
        padded_audio = audio[rand_onset:rand_onset + target_len]
        onset_s = -round(rand_onset / fs, 3)
        offset_s = 10.0

    else:
        padded_audio = audio
        onset_s = 0.000
        offset_s = 10.0

    return padded_audio, onset_s, offset_s

def process_labels(df, onset):
    df["onset"] = df["onset"] + onset 
    df["offset"] = df["offset"] + onset
        
    df["onset"] = df.apply(lambda x: max(0, x["onset"]), axis=1)
    df["offset"] = df.apply(lambda x: min(10, x["offset"]), axis=1)

    df_new = df[(df.onset < df.offset)]
    
    return df_new.drop_duplicates()

def read_audio(file, pad_to, multisrc = False, random_channel = False):
    mixture, fs = torchaudio.load(file)
    
    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    onset_s = None
    if pad_to is not None:
        mixture, onset_s, offset_s = pad_audio(mixture, pad_to, fs)

    mixture = mixture.float()

    return file, mixture, onset_s, offset_s

# data augmentation for the strongly labeled audio
class MixUp(torch.nn.Module):
    r"""
    Modified version of timm.data.mixup.Mixup. (https://timm.fast.ai/mixup_cutmix)

    Args:
        specgrams (Tensor): Real spectrograms (batch, channel, freq, time)
        num_classes (float): total number of classes
        labels (Tensor): One-hot encodings (batch, n_class) or (batch, time, n_class)
        alpha (float): Beta distribution concentration parameter 
        mix_prob (float): Probability of performing mixup
        mode (str): How to apply mixup params. Per "batch" or "elem"

    Returns:
        Tensor: Masked spectrograms of dimensions (batch, channel, freq, time)
        Tensor: Masked one-hot encodings of dimensions (batch, n_class) or (batch, time, n_class)
    """
    def __init__(self,
                 num_classes: int,
                 alpha: float = 0.2,
                 mix_prob: float = 0.5,
                 mode: str = 'elem')-> None:
        super(MixUp, self).__init__()

        assert mode in ['elem', 'batch'], f"mode: {mode} not implemented. choice in ['elem', 'batch']"
        self.num_classes = num_classes
        self.mix_prob = mix_prob
        self.mode = mode

        self.beta_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _params_per_elem(self, batch_size: int)-> Tensor:
        lam = torch.ones(batch_size, dtype=torch.float32)
        lam_mix = self.beta_dist.sample(sample_shape = (batch_size, )).to(torch.float32)[:, 0]
        lam = torch.where(torch.rand(batch_size) < self.mix_prob, lam_mix, lam)
        return lam

    def _params_per_batch(self)-> float:
        lam = 1.
        if torch.rand() < self.mix_prob:
            lam_mix = self.beta_dist.sample().to(torch.float32)

        lam = lam_mix
        return lam

    def apply_transform(self, x: Tensor, lam: Tensor)-> Tensor:
        assert len(lam.size()) == 1, f"dimension size of argument lam should be 1 (but got {len(lam.size())})"
        
        for _ in range(len(x.size())-1):
            lam = lam.unsqueeze(1)
        
        x = lam * x + (1 - lam) * x.flip(0)

        return x

    def forward(self, spectrogram: Tensor, label: Tensor = None)-> Union[Tensor, Tuple]:
        if not isinstance(spectrogram, Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(spectrogram)))

        batch_size = spectrogram.size(0)
        device = spectrogram.device

        if self.mode == 'elem':
            lam = self._params_per_elem(batch_size)

        elif self.mode == "batch":
            lam = self._params_per_batch()

        if (lam == 1.).all():
            if label is None:
                return spectrogram
            else:
                return spectrogram, label
        
        lam = lam.to(device)

        mixed_specgrams = self.apply_transform(spectrogram, lam)

        if label is not None:
            assert (label.sum(dim = -1) == 1).all() and len(label.size()) in [2, 3], f"label should be one-hot-encoded"

            mixed_labels = self.apply_transform(label, lam)
            mixed_labels = torch.clamp(mixed_labels, min=0, max=1)

            return mixed_specgrams, mixed_labels

        return mixed_specgrams


class SpecAugment(torch.nn.Module):
    def __init__(self, n_time = 1, n_freq = 1, time_mask_param = 5, freq_mask_param = 5, iid_masks = True, p = 0.5):
        super(SpecAugment, self).__init__()
        self.n_time = n_time
        self.n_freq = n_freq
        self.p = p
        self.time_mask_param = time_mask_param
        self.FreqMask = T.FrequencyMasking(freq_mask_param = freq_mask_param, iid_masks= iid_masks)

    def apply_transform(self, spectrogram: torch.Tensor, label = None):
        for i in range(self.n_time):
            if label is not None:
                spectrogram, label = self.time_masking(spectrogram, label, mask_value = 0.)
            else:
                spectrogram = self.time_masking(spectrogram)

        for j in range(self.n_freq):
            spectrogram = self.FreqMask(spectrogram)

        if label is not None:
            return spectrogram, label
        
        return spectrogram

    def forward(self, spectrogram: torch.Tensor, label: torch.Tensor = None):
        mask = torch.rand(size = (len(spectrogram),)) < self.p

        if label is not None:
            spectrogram[mask], label[mask] = self.apply_transform(spectrogram[mask], label[mask])

            return spectrogram, label
        else:
            spectrogram[mask] = self.apply_transform(spectrogram[mask])

            return spectrogram
    
    def time_masking(self, spectrograms, label = None, axis = 2, mask_value = 0.0):
        # spectrograms (batch_size, channel, freq, time)
        # label (batch_size, time, class)
        mask_param = self._get_mask_param(self.time_mask_param, spectrograms.shape[axis])

        device = spectrograms.device
        dtype = spectrograms.dtype

        value = torch.rand(spectrograms.shape[:2], device=device, dtype=dtype) * mask_param
        min_value = torch.rand(spectrograms.shape[:2], device=device, dtype=dtype) * (spectrograms.size(axis) - value)

        # Create broadcastable mask
        mask_start = min_value.long()[..., None, None]
        mask_end = (min_value + value).long()[..., None, None]
        mask = torch.arange(0, spectrograms.size(axis), device=device, dtype=dtype)
        print(mask_start, mask_end)

        # Per batch example masking
        spectrograms = spectrograms.transpose(axis, -1)
        spectrograms = spectrograms.masked_fill((mask >= mask_start) & (mask < mask_end), mask_value)
        spectrograms = spectrograms.transpose(axis, -1)
        
        if label is not None:
            mask_start = (mask_start.squeeze(1)/spectrograms.size(axis)*label.size(axis-1)).long()
            mask_end = (mask_end.squeeze(1)/spectrograms.size(axis)*label.size(axis-1)).long()
            mask = torch.arange(0, label.size(axis-2), device=device, dtype=dtype)

            label = label.transpose(axis-2, -1)
            label = label.masked_fill((mask >= mask_start) & (mask < mask_end), mask_value)
            label = label.transpose(axis-2, -1)

            return spectrograms, label

        return spectrograms

    def _get_mask_param(self, mask_param: int, axis_length: int, p: float = 1.0) -> int:
        if p == 1.0:
            return mask_param
        else:
            return min(mask_param, int(axis_length * p))


class TimeShift(torch.nn.Module):
    def __init__(self, max_shift):
        super(TimeShift, self).__init__()
        self.max_shift = max_shift  # Maximum shift in number of time steps

    def forward(self, spectrogram, labels = None):
        """
        Shifts the spectrogram along the time axis with circular boundary conditions and adjusts labels accordingly.

        Parameters:
            spectrogram (Tensor): The input spectrogram, shape (batch_size, channels, freq_bins, time_steps)
            labels (Tensor): One-hot encoded labels, shape (batch_size, time_steps, n_class)

        Returns:
            Tuple[Tensor, Tensor]: The shifted spectrogram and adjusted labels
        """
        shift_amount = np.random.randint(-self.max_shift, self.max_shift + 1)

        # Shift spectrogram and labels with circular boundary conditions
        shifted_spectrogram = torch.roll(spectrogram, shifts=shift_amount, dims=-1)

        if labels is not None:
            shift_amount = int(shift_amount/spectrogram.size(-1)*labels.size(-2))
            adjusted_labels = torch.roll(labels, shifts=shift_amount, dims=1)

            return shifted_spectrogram, adjusted_labels

        return shifted_spectrogram


def batched_decode_preds(
    strong_preds, filenames, encoder, thresholds=[0.5], median_filter=7, pad_indx=None,
):
    """ Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary

    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {}
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches
        audio_id = Path(filenames[j]).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[j]
        if pad_indx is not None:
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]
        c_scores = c_scores.transpose(0, 1).detach().cpu().numpy()
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores)+1)),
            event_classes=encoder.labels,
        )
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores)+1)),
            event_classes=encoder.labels,
        )
        for c_th in thresholds:
            pred = c_scores > c_th
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat([prediction_dfs[c_th], pred], ignore_index=True)

    return scores_raw, scores_postprocessed, prediction_dfs


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def psds_results(psds_obj):
    """ Compute psds scores
    Args:
        psds_obj: psds_eval.PSDSEval object with operating points.
    Returns:
    """
    try:
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (0, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (1, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=1, max_efpr=100)
        print(f"\nPSD-Score (0, 1, 100): {psds_score.value:.5f}")
    except psds_eval.psds.PSDSEvalError as e:
        print("psds did not work ....")
        raise EnvironmentError


def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """ Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """ Calculate SegmentBasedMetrics given a reference and estimated dataframe

        Args:
            reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                reference events
            estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                estimated events to be compared with reference
            time_resolution: float, the time resolution of the segment based metric
        Returns:
             sed_eval.sound_event.SegmentBasedMetrics with the scores
        """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric


def compute_sed_eval_metrics(predictions, groundtruth):
    """ Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )

    return metric_event, metric_segment


def compute_per_intersection_macro_f1(
    prediction_dfs,
    # ground_truth_file,
    gt,
    durations,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
):
    """ Compute F1-score per intersection, using the defautl
    Args:
        prediction_dfs: dict, a dictionary with thresholds keys and predictions dataframe
        ground_truth_file: pd.DataFrame, the groundtruth dataframe
        durations_file: pd.DataFrame, the duration dataframe
        dtc_threshold: float, the parameter used in PSDSEval, percentage of tolerance for groundtruth intersection
            with predictions
        gtc_threshold: float, the parameter used in PSDSEval percentage of tolerance for predictions intersection
            with groundtruth
        gtc_threshold: float, the parameter used in PSDSEval to know the percentage needed to count FP as cross-trigger

    Returns:

    """
    # gt = pd.read_csv(ground_truth_file, sep="\t")
    # durations = pd.read_csv(durations_file, sep="\t")
    
    durations_df = {'filename': [], 'duration': []}
    for k, v in durations.items():
        durations_df['filename'].append(k)
        durations_df['duration'].append(v)

    durations_df = pd.DataFrame(durations_df)

    psds = PSDSEval(
        ground_truth=gt,
        metadata=durations_df,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )
    psds_macro_f1 = []
    for threshold in prediction_dfs.keys():
        if not prediction_dfs[threshold].empty:
            threshold_f1, _ = psds.compute_macro_f_score(prediction_dfs[threshold])
        else:
            threshold_f1 = 0
        if np.isnan(threshold_f1):
            threshold_f1 = 0.0
        psds_macro_f1.append(threshold_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1


def compute_psds_from_operating_points(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):

    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.join(
            save_dir,
            f"predictions_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}",
        )
        os.makedirs(pred_dir, exist_ok=True)
        for k in prediction_dfs.keys():
            prediction_dfs[k].to_csv(
                os.path.join(pred_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )

        filename = (
            f"PSDS_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}"
            f"_ct{alpha_ct}_st{alpha_st}_max{max_efpr}_psds_eval.png"
        )
        plot_psd_roc(
            psds_score,
            filename=os.path.join(save_dir, filename),
        )

    return psds_score.value


def compute_psds_from_scores(
    scores,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    num_jobs=1,
    save_dir=None,
):
    psds, psd_roc, single_class_rocs, *_ = sed_scores_eval.intersection_based.psds(
        scores=scores, ground_truth=ground_truth_file,
        audio_durations=durations_file,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        max_efpr=max_efpr, num_jobs=num_jobs,
    )
    if save_dir is not None:
        scores_dir = os.path.join(save_dir, "scores")
        sed_scores_eval.io.write_sed_scores(scores, scores_dir)
        filename = (
            f"PSDS_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}"
            f"_ct{alpha_ct}_st{alpha_st}_max{max_efpr}_sed_scores_eval.png"
        )
        sed_scores_eval.utils.visualization.plot_psd_roc(
            psd_roc,
            filename=os.path.join(save_dir, filename),
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold, alpha_ct=alpha_ct,
            alpha_st=alpha_st, unit_of_time='hour',  max_efpr=max_efpr,
            psds=psds,
        )
    return psds


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """ Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    # gt = pd.read_csv(ground_truth, sep="\t")
    gt = ground_truth

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures