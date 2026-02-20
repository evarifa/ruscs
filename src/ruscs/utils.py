import yaml
import numpy as np
import sys
import logging
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mutual_info_score
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.relativeSeconds = int(record.relativeCreated / 1000)
        return super().format(record)
    
class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.rstrip() != "":
            logging.log(self.level, message.rstrip())
    def flush(self):
        pass

class InlineListDumper(yaml.Dumper):
    def represent_dict(self, data):
        return self.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data, flow_style=False
        )
    def represent_list(self, data):
        return self.represent_sequence(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, data, flow_style=True
        )

def init_logging(output, name = None, unit="sec"):
    if name:
        namelog = f'{output}/logfile_{name}.log'
    else:
        today = datetime.now().strftime("%Y%m%d")
        namelog = f'{output}/logfile_{today}.log'
    handler = logging.FileHandler(namelog)
    formatter = CustomFormatter(fmt="%(asctime)s [%(relativeSeconds)d s] %(message)s",
                                datefmt='%m/%d/%Y %I:%M:%S %p')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    sys.stdout = LoggerWriter(logging.INFO)
    sys.stderr = LoggerWriter(logging.ERROR)
    return

def create_symbolic_array(symbolic_data_y, symbolic_data_z, networksize, niter, maxiter):
    sym_array = np.zeros((niter, networksize, maxiter))
    for i in range(niter):
        for j in range(maxiter):
            for k in symbolic_data_y[i][j]:
                sym_array[i, k-1, j] = 1 # 1 spreader
            for l in symbolic_data_z[i][j]:
                sym_array[i, l-1, j] = 2 # 2 stifler
    return sym_array


def linear_increase(total_length, y_min, y_max, steps_wait, steps_increase):
    res = np.zeros(total_length)
    for i in range(total_length):
        if i < steps_wait:
            res[i] = y_min
        elif i < (steps_increase + steps_wait):
            res[i] = y_min + (i - steps_wait) * (y_max - y_min) / (steps_increase)
        else:
            res[i] = y_max
    return res


def rolling_mutualinfo_neighbors_pulse(data, window_size, ininetworksize, neighbors, pulse_vector, kini):
    """Compute mutual info."""
    index_pulse = np.where(pulse_vector != 0)[0]
    max_index_pulse = np.max(index_pulse)
    series_length = data.shape[1]
    network_size = data.shape[0]
    mi_values = np.zeros(series_length) # [len(timeseries)]
    mi_values[0:window_size - 1] = np.nan # Pad the start with NaN to match the input length
    for i in range(series_length - window_size + 1):
        if i < max_index_pulse + window_size:
            window = data[:ininetworksize, i:i + window_size] # network, window 
            a = np.random.randint(ininetworksize)
            b = np.random.randint(len(neighbors[a][:kini]))
        else:
            window = data[:, i:i + window_size] # network, window 
            a = np.random.randint(network_size)
            b = np.random.randint(len(neighbors[a]))
        mi_values[i + window_size - 1] = mutual_info_score(window[a, :], window[neighbors[a][b]-1, :])
    return mi_values


def rolling_mutualinfo_random_pulse(data, window_size, ininetworksize, pulse_vector):
    """Compute mutual info."""
    index_pulse = np.where(pulse_vector != 0)[0]
    max_index_pulse = np.max(index_pulse)
    series_length = data.shape[1]
    network_size = data.shape[0]
    mi_values = np.zeros(series_length) # [len(timeseries)]
    mi_values[0:window_size - 1] = np.nan # Pad the start with NaN to match the input length
    for i in range(series_length - window_size + 1):
        if i < max_index_pulse + window_size:
            window = data[:ininetworksize, i:i + window_size] # network, window 
            a = np.random.randint(ininetworksize)
            b = np.random.randint(ininetworksize)
        else:
            window = data[:, i:i + window_size] # network, window 
            a = np.random.randint(network_size)
            b = np.random.randint(network_size)
        mi_values[i + window_size - 1] = mutual_info_score(window[a, :], window[b, :])
    return mi_values
    

def rolling_autocorrelation(time_series, window_size, lag):
    autocorr_values = np.zeros((lag, len(time_series))) # [nlag, len(timeseries)]
    autocorr_values[:, 0:window_size - 1] = np.nan # Pad the start with NaN to match the input length
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        autocorr = acf(window, nlags = lag)
        if len(autocorr) > lag: # nlag + 0-lag
            autocorr_values[:, i + window_size - 1] = autocorr[1:lag+1]
        else:
            autocorr_values[:, i + window_size - 1] = np.append(autocorr, np.repeat(np.nan, lag-len(autocorr)))
    return autocorr_values



def rolling_mutualinfo_random(data, window_size):
    series_length = data.shape[1]
    network_size = data.shape[0]
    mi_values = np.zeros(series_length) # [len(timeseries)]
    mi_values[0:window_size - 1] = np.nan # Pad the start with NaN to match the input length
    for i in range(series_length - window_size + 1):
        window = data[:, i:i + window_size] # network, window 
        a = np.random.randint(network_size)
        b = np.random.randint(network_size)
        mi_values[i + window_size - 1] = mutual_info_score(window[a, :], window[b, :])
    return mi_values

def rolling_mutualinfo_neighbors(data, window_size, neighbors):
    series_length = data.shape[1]
    network_size = data.shape[0]
    mi_values = np.zeros(series_length) # [len(timeseries)]
    mi_values[0:window_size - 1] = np.nan # Pad the start with NaN to match the input length
    for i in range(series_length - window_size + 1):
        window = data[:, i:i + window_size] # network, window 
        a = np.random.randint(network_size)
        b = np.random.randint(len(neighbors[a]))
        mi_values[i + window_size - 1] = mutual_info_score(window[a, :], window[neighbors[a][b]-1, :])
    return mi_values

    

