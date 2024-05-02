# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Dataset Collections

import numpy as np
import torch
import logging
import os
import sys
import h5py
import csv
import time
import random
import json
from collections import defaultdict
from datetime import datetime
from torch.utils.data import Dataset
from utils import int16_to_float32
from glob import glob
import pandas as pd
import torchaudio

from utils import read_audio, process_labels

class StronglyAnnotatedSet(Dataset):
    def __init__(self, audio_folder, tsv_entries, encoder, config, pad_to=10, resolution = 0.320, eval_mode = False):
        self.encoder = encoder
        self.config = config
        self.fs = self.config.sample_rate
        self.pad_to = pad_to * self.fs
        self.eval_mode = eval_mode

        self._init_examples(tsv_entries, audio_folder, resolution = 0.320)

        if not self.eval_mode:
            self.generate_queue()

    def _init_examples(self, tsv_entries, audio_folder, resolution = 0.320):
        tsv_entries = tsv_entries.dropna()
        self.tsv_entries = []
        examples = {}
        self.class_set = []
        self.audio_durations = {}
        self.ground_truth = {}
        self.idx = defaultdict(list)

        for _, r in tsv_entries.iterrows():
            if r["end_time_seconds"] - r["start_time_seconds"] < resolution:
                continue

            
            if r["segment_id"] not in examples.keys():
                examples[r["segment_id"]] = {"mixture": os.path.join(audio_folder, r["segment_id"]),
                                             "events": []}
                self.ground_truth[os.path.splitext(r["segment_id"])[0]] = []

            examples[r["segment_id"]]["events"].append(
                {"event_label": r["label"],
                    "onset": r["start_time_seconds"],
                    "offset": r["end_time_seconds"],}
                )
            self.ground_truth[os.path.splitext(r["segment_id"])[0]].append([r["start_time_seconds"], r["end_time_seconds"], r["label"]])
            
            class_idx = self.encoder.labels.index(r["label"])
            self.idx[class_idx].append(r["segment_id"])

            if class_idx not in self.class_set:
                self.class_set.append(class_idx)

            if os.path.splitext(r["segment_id"])[0] not in self.audio_durations.keys():
                metadata = torchaudio.info(os.path.join(audio_folder, r["segment_id"]))
                self.audio_durations[os.path.splitext(r["segment_id"])[0]] = metadata.num_frames / metadata.sample_rate

            self.tsv_entries.append(r)

        self.tsv_entries = pd.DataFrame(self.tsv_entries, columns=tsv_entries.columns)

        # we construct a dictionary for each example
        self.examples = examples
        self.examples_list = list(examples.keys())

        self.total_size = len(self.examples_list)
        self.classes_num = len(self.class_set)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        if not self.eval_mode:
            c_ex = self.examples[self.queue[index]]
        else:
            c_ex = self.examples[self.examples_list[index]]

        audio_name, mix_waveform, onset_s, offset_s = read_audio(c_ex["mixture"], self.pad_to)

        # labels
        labels = c_ex["events"]
        
        # to steps
        labels_df = pd.DataFrame(labels)
        labels_df = process_labels(labels_df, onset_s)
        
        # check if labels exists:
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()

        else:
            strong = self.encoder.encode_strong_df(labels_df)
            strong = torch.from_numpy(strong).float()
        print(index, labels_df)
        data_dict = {
                "audio_name": audio_name,
                "waveform": mix_waveform, # (1, time)
                "target": strong, # (time, classes)
                "onoff": [max(onset_s, 0.)/self.pad_to*self.fs, min(offset_s, 10.)/self.pad_to*self.fs],
            }

        return data_dict

    def generate_queue(self):
        self.queue = []      
        if self.config.debug:
            self.total_size = 1000

        if self.config.balanced_data:
            if self.config.class_filter is not None:
                class_set = self.config.class_filter[:]
            else:
                class_set = self.class_set

            if self.config.enable_token_label:
                while len(self.queue) < self.total_size * 2:
                    random.shuffle(class_set)
                    self.queue += [self.idx[d][random.randint(0, len(self.idx[d]) - 1)] for d in class_set]

                self.queue = self.queue[:self.total_size * 2]
                self.queue = [[self.queue[i],self.queue[i+1]] for i in range(0, self.total_size * 2, 2)]
                assert len(self.queue) == self.total_size, "generate data error!!" 

            else:
                while len(self.queue) < self.total_size:
                    random.shuffle(class_set)
                    self.queue += [self.idx[d][random.randint(0, len(self.idx[d]) - 1)] for d in class_set]

                self.queue = self.queue[:self.total_size]
        else:
            self.queue = [*range(self.total_size)]
            random.shuffle(self.queue)
        
        logging.info("queue regenerated:%s" %(self.queue[:5]))
    
# For AudioSet
class SEDDataset(Dataset):
    def __init__(self, index_path, idc, config, eval_mode = False):
        """
        Args:
           index_path: the link to each audio
           idc: npy file, the number of samples in each class, computed in main
           config: the config.py module 
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        self.config = config
        self.fp = h5py.File(index_path, "r")
        self.idc = idc
        self.total_size = len(self.fp["audio_name"])
        self.classes_num = config.classes_num
        self.eval_mode = eval_mode
        self.shift_max = config.shift_max
        if (config.enable_label_enhance) and (not eval_mode):
            self.class_map = np.load(config.class_map_path, allow_pickle = True)

        if not eval_mode:
            self.generate_queue()
        else:
            if self.config.debug:
                self.total_size = 1000
            self.queue = []
            for i in range(self.total_size):
                target = self.fp["target"][i]
                if np.sum(target) > 0:
                    self.queue.append(i)
            self.total_size = len(self.queue)
        logging.info("total dataset size: %d" %(self.total_size))
        logging.info("class num: %d" %(self.classes_num))

    def time_shifting(self, x):
        frame_num = len(x)
        shift_len = random.randint(0, self.shift_max - 1)
        new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis = 0)
        return new_sample 

    def generate_queue(self):
        self.queue = []      
        if self.config.debug:
            self.total_size = 1000
        if self.config.balanced_data:
            if self.config.enable_token_label:
                while len(self.queue) < self.total_size * 2:
                    if self.config.class_filter is not None:
                        class_set = self.config.class_filter[:]
                    else:
                        class_set = [*range(self.classes_num)]
                    random.shuffle(class_set)
                    self.queue += [self.idc[d][random.randint(0, len(self.idc[d]) - 1)] for d in class_set]
                self.queue = self.queue[:self.total_size * 2]
                self.queue = [[self.queue[i],self.queue[i+1]] for i in range(0, self.total_size * 2, 2)]
                assert len(self.queue) == self.total_size, "generate data error!!" 
            else:
                while len(self.queue) < self.total_size:
                    if self.config.class_filter is not None:
                        class_set = self.config.class_filter[:]
                    else:
                        class_set = [*range(self.classes_num)]
                    random.shuffle(class_set)
                    self.queue += [self.idc[d][random.randint(0, len(self.idc[d]) - 1)] for d in class_set]
                self.queue = self.queue[:self.total_size]
        else:
            self.queue = [*range(self.total_size)]
            random.shuffle(self.queue)
        
        logging.info("queue regenerated:%s" %(self.queue[-5:]))

    def crop_wav(self, x):
        crop_size = self.config.crop_size
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos:crop_pos + crop_size]

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "hdf5_path": str,
            "index_in_hdf5": int,
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        s_index = self.queue[index]
        if (not self.eval_mode) and (self.config.enable_token_label):
            audio_name = self.fp["audio_name"][s_index[0]].decode()
            hdf5_path = [
                self.fp["hdf5_path"][s_index[0]], 
                self.fp["hdf5_path"][s_index[1]]
                ]
            r_idx = [
                self.fp["index_in_hdf5"][s_index[0]],
                self.fp["index_in_hdf5"][s_index[1]]
                ]
            target = [
                self.fp["target"][s_index[0]].astype(np.float32),
                self.fp["target"][s_index[1]].astype(np.float32)
            ]
            waveform = []
            with h5py.File(hdf5_path, "r") as f:
                waveform.append(int16_to_float32(f["waveform"][r_idx[0]]))
            with h5py.File(hdf5_path, "r") as f:
                waveform.append(int16_to_float32(f["waveform"][r_idx[1]]))
            mix_sample = int(len(waveform[1]) * random.uniform(self.config.token_label_range[0],self.config.token_label_range[1]))
            mix_position = random.randint(0, len(waveform[1]) - mix_sample - 1)
            mix_waveform = np.concatenate(
                [waveform[0][:mix_position], 
                waveform[1][mix_position:mix_position+mix_sample],
                waveform[0][mix_position+mix_sample:]],
                axis=0
            )
            mix_target = np.concatenate([
                np.tile(target[0],(mix_position,1)),
                np.tile(target[1], (mix_sample, 1)),
                np.tile(target[0], (len(waveform[0]) - mix_position - mix_sample, 1))],
                axis=0
            ) 
            assert len(mix_waveform) == len(waveform[0]),"length of the mix waveform error!!"
            data_dict = {
                "audio_name": audio_name,
                "waveform": mix_waveform,
                "target": mix_target
            }
        else:
            audio_name = self.fp["audio_name"][s_index].decode()
            hdf5_path = self.fp["hdf5_path"][s_index].decode()
            # replace("/home/tiger/DB/knut/data/audioset", self.config.dataset_path)
            r_idx = self.fp["index_in_hdf5"][s_index]
            target = self.fp["target"][s_index].astype(np.float32)
            with h5py.File(hdf5_path, "r") as f:
                waveform = int16_to_float32(f["waveform"][r_idx])
            # Time shift
            if (self.config.enable_time_shift) and (not self.eval_mode):
                waveform = self.time_shifting(waveform)
            # Label Enhance
            if (self.config.crop_size is not None) and (not self.eval_mode):
                waveform = self.crop_wav(waveform)
            # the label enhance rate is fixed 0.5
            if (self.config.enable_label_enhance) and (not self.eval_mode) and random.random() < 0.5:
                kidx = np.where(target)[0]
                for k in kidx:
                    for add_key in self.class_map[k][1]:
                        target[add_key] = 1.0
                    if len(self.class_map[k][2]) > 0:
                        add_key = random.choice(self.class_map[k][2])
                        target[add_key] = 1.0
      
            data_dict = {
                "hdf5_path": hdf5_path,
                "index_in_hdf5": r_idx,
                "audio_name": audio_name,
                "waveform": waveform,
                "target": target
            }
        return data_dict

    def __len__(self):
        return self.total_size

# For ESC dataset
class ESC_Dataset(Dataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.dataset = self.dataset[self.config.esc_fold]
        else:
            temp = []
            for i in range(len(self.dataset)):
                if i != config.esc_fold:
                    temp += list(self.dataset[i]) 
            self.dataset = temp           
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)
        logging.info("queue regenerated:%s" %(self.queue[-5:]))


    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]
        data_dict = {
            "audio_name": self.dataset[p]["name"],
            "waveform": np.concatenate((self.dataset[p]["waveform"],self.dataset[p]["waveform"])),
            "real_len": len(self.dataset[p]["waveform"]) * 2,
            "target": self.dataset[p]["target"]
        }
        return data_dict

    def __len__(self):
        return self.total_size


# For Speech Command V2 dataset
class SCV2_Dataset(Dataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config
        self.eval_mode = eval_mode
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)
        logging.info("queue regenerated:%s" %(self.queue[-5:]))


    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]

        waveform  = self.dataset[p]["waveform"]
        while len(waveform) < self.config.clip_samples:
            waveform = np.concatenate((waveform, waveform))
        waveform = waveform[:self.config.clip_samples]

        target = np.zeros(self.config.classes_num).astype(np.float32)
        target[int(self.dataset[p]["target"])] = 1.
        data_dict = {
            "audio_name": self.dataset[p]["name"],
            "waveform": waveform,
            "real_len": len(waveform),
            "target": target
        }
        return data_dict

    def __len__(self):
        return self.total_size

# For DeSED dataset in DACASE 2020/2021
class DESED_Dataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.total_size = len(dataset)
        logging.info("total dataset size: %d" %(self.total_size))

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
        }
        """
        real_len = len(self.dataset[index]["waveform"])
        if real_len < self.config.clip_samples:
            zero_pad = np.zeros(self.config.clip_samples - real_len)
            waveform = np.concatenate([self.dataset[index]["waveform"], zero_pad])
        else:
            waveform = self.dataset[index]["waveform"][:self.config.clip_samples]
        data_dict = {
            "audio_name": self.dataset[index]["audio_name"],
            "waveform": int16_to_float32(waveform),
            "real_len": real_len
        }
        return data_dict

    def __len__(self):
        return self.total_size
