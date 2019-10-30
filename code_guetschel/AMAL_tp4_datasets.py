import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from math import ceil
import string
import unicodedata


##############################################################################
#### Temperatures
##############################################################################

class DatasetTempClassif(Dataset):
    def __init__(self, get_fun, indexes, cities):
        self.get_fun = get_fun
        self.num_cities = len(cities)
        self.length = indexes.shape[0] * self.num_cities
        self.indexes = indexes
    def __getitem__(self, idx):
        i_city = idx % self.num_cities
        i_idx = idx // self.num_cities
        return self.get_fun(i_city, *self.indexes[i_idx])
    def __len__(self):
        return self.length

class DatasetTempForecast(Dataset):
    def __init__(self, get_fun, indexes, cities, one_city=False):
        self.get_fun = get_fun
        self.num_cities = len(cities)
        self.one_city = one_city
        if one_city:
            self.length = indexes.shape[0] * self.num_cities
        else:
            self.length = indexes.shape[0]
        self.indexes = indexes
    def __getitem__(self, idx):
        if self.one_city:
            i_city = idx % self.num_cities
            i_idx = idx // self.num_cities
        else:
            i_city = None
            i_idx = idx
        seq,_ = self.get_fun(i_city, *self.indexes[i_idx])
        return seq[:-1], seq[1:]
    def __len__(self):
        return self.length

class DataHolder:
    """
    The train/val/test split is done once and for all in __init__()
    over somme continious sequences of of size `mega_sample_length`

    Then get_datasets() generates somme `torch.utils.data.Dataset`
    of fixed length sequences by cuting the mega samples
    """
    def __init__(self, mega_sample_length, train_val_test_split=(.8, .1, .1), path="./tempAMAL_train.csv", seed=12, max_num_city=None):
        self.sample_length = mega_sample_length
        with open(path, 'r') as f:
            l0  = f.readline()
        if max_num_city is None:
            self.num_cities = len(l0.split(',')) - 1
        else:
            self.num_cities = max_num_city
        dtype = dict(zip(range(self.num_cities+1), ['str']+['float32' for _ in range(self.num_cities)]))
        datetime_parser = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
        raw_ds = pd.read_csv(path, dtype=dtype, parse_dates=['datetime'], date_parser=datetime_parser)

        # separe time and data columns :
        self.timestamps = raw_ds.iloc[:, :1]
        self.dataset    = raw_ds.iloc[:, 1:self.num_cities+1]
        # normalize time :
        months = self.timestamps['datetime'].map(lambda x:x.month)
        days   = self.timestamps['datetime'].map(lambda x:x.day)
        times  = self.timestamps['datetime'].map(lambda x:x.hour)
        self.timestamps['month'] = ((months-months.min()) / (months.max()-months.min())).astype('float32')
        self.timestamps['day'  ] = ((days  -  days.min()) / (  days.max()-  days.min())).astype('float32')
        self.timestamps['time' ] = ((times - times.min()) / ( times.max()- times.min())).astype('float32')
        # normalze data :
        self.dataset = pd.DataFrame((self.dataset.values - self.dataset.min().min()) / (self.dataset.max().max() - self.dataset.min().min()))
        # remove nan by extending last value :
        self.dataset = self.dataset.fillna(method='ffill')

        self.cities = self.dataset.columns

        # split the parts of the dataset :
        dataset_length = len(self.dataset.iloc[:,0])
        intervals = []
        for i in range(ceil(float(dataset_length)/mega_sample_length)):
            intervals.append( (i*mega_sample_length, min(dataset_length, (i+1)*mega_sample_length)) )
        intervals = np.array(intervals, dtype=np.int64)
        indexes = np.arange(intervals.shape[0])
        self.splited_samples = []
        for i,p in enumerate(train_val_test_split):
            idx = np.random.choice(indexes, int(len(intervals) * p ))
            self.splited_samples.append(intervals[idx])
            indexes = indexes[np.where(~np.isin(indexes,idx))]
        if indexes.shape[0] != 0:
            self.splited_samples[0] = np.concatenate([self.splited_samples[0], intervals[indexes,:]], axis=0)

    def get_num_classes(self):
        return len(self.cities)

    def get_sample(self, i_city, idx0, idx1):
        timestamps = self.timestamps.iloc[idx0 : idx1, 1:]
        if i_city is None:
            sample = self.dataset.iloc[idx0 : idx1, :].values
        else:
            sample = np.expand_dims(self.dataset.iloc[idx0 : idx1, i_city].values, axis=-1)
        return np.concatenate([sample, timestamps], axis=1), i_city

    @staticmethod
    def cut_sample(sample, size, stride, drop_last=True):
        idx0, idx1 = sample
        return np.array([(i, min(idx1, i+size)) for i in range(idx0, idx1, stride) if i+size<=idx1 or not drop_last])

    @staticmethod
    def get_subsamples(samples, sample_size, stride):
        return np.concatenate([DataHolder.cut_sample(s, sample_size, stride, drop_last=True) for s in samples])

    def get_datasets_classif(self, sample_size, strides):
        if sample_size>self.sample_length:
            raise AttributeError
        sub_samlpes = [self.get_subsamples(s, sample_size, stride) for s,stride in zip(self.splited_samples, strides)]
        return [DatasetTempClassif(self.get_sample, s, self.cities) for s in sub_samlpes]

    def get_datasets_forecast(self, sample_size, strides, one_city=False):
        if sample_size>self.sample_length:
            raise AttributeError
        sub_samlpes = [self.get_subsamples(s, sample_size, stride) for s,stride in zip(self.splited_samples, strides)]
        return [DatasetTempForecast(self.get_sample, s, self.cities, one_city=one_city) for s in sub_samlpes]


##############################################################################
#### Speach
##############################################################################

LETTRES = string.ascii_letters+string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1, len(LETTRES) + 1 ), LETTRES) )
id2lettre[0] = '' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys() ) )

class DatasetSpeach(Dataset):
    def __init__(self, data, sample_size, stride):
        super().__init__()
        self.data = data
        self.sample_size = sample_size
        self.stride = stride
        self.length = (data.shape[0] - sample_size) // stride
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        d = self.data[idx*self.stride : idx*self.stride + self.sample_size]
        return d[:-1], d[1:]

class SpeachHolder:
    def __init__(self, path='./trump_full_speech.txt', train_val_test_split=(.8, .1, .1)):
        with open(path, "r") as f:
            text = f.readlines()
        text = " ".join(text)
        text_len = len(text)
        parts = []
        last = 0
        for p in train_val_test_split:
            next = last + min(int(p*text_len), text_len)
            parts.append(text[last:next])
            last = next
        with torch.no_grad():
            self.parts = [self.string2code(p) for p in parts]

    def get_datasets(self, sample_size, strides):
        return [DatasetSpeach(part, sample_size, stride) for part,stride in zip(self.parts, strides)]

    @staticmethod
    def normalize(s) :
        return ''.join(c for c in unicodedata.normalize('NFD', s ) if c in LETTRES)
    @staticmethod
    def string2code(s) :
        return torch.tensor([lettre2id[c] for c in SpeachHolder.normalize(s)] )
    @staticmethod
    def code2string ( t ) :
        if type(t) != list :
            t = t.tolist()
        return ''.join(id2lettre[i] for i in t)
    @staticmethod
    def get_num_classes():
        return len(lettre2id.keys())
