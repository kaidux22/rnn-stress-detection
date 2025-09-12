import neurokit2 as nk
from math import floor, ceil
import pickle
import numpy as np
import warnings

import downloader as dl

class Signal:
  """
  name - название сигнала
  sampling - частота дискретизации (в Гц)
  data - сам массив данных
  """
  def __init__(self, name, sampling, data):
      self.__name = name
      self.__sampling = sampling

      with warnings.catch_warnings():
          warnings.filterwarnings('ignore', module='neurokit2')
          
          if 'EDA' in name:
              self.__data = {"Clean": nk.eda_clean(data, sampling_rate=sampling)}
              components = nk.eda_phasic(self.__data['Clean'], sampling_rate=sampling)
              self.__data['Phasic'], self.__data['Tonic'] = components['EDA_Phasic'].values, components['EDA_Tonic'].values
              self.__data['Peaks'], _ = nk.eda_peaks(self.__data['Clean'], sampling_rate=sampling)
              self.__data['Peaks'] = self.__data['Peaks']["SCR_Peaks"].values
          elif "BVP" in name:
              self.__data, _ = nk.ppg_process(data, sampling_rate=sampling)
          elif "ECG" in name:
              self.__data, _ = nk.ecg_process(data, sampling_rate=sampling)
          else:
              self.__data = data

  def __str__(self):
    return f"Name: {self.__name}, Count: {len(self.__data)}, {self.__sampling} Hz"

  def get_freq(self):
    return self.__sampling

  def get_name(self):
    return self.__name

  def get_data(self):
    return self.__data

  def get_data_segment(self, time_begin, time_end, origin):
    if "EDA" in self.__name:
      return {
          "Clean": self.__data["Clean"][floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Tonic": self.__data["Tonic"][floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Phasic": self.__data["Phasic"][floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Peaks": self.__data["Peaks"][floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1]
      }
    elif "BVP" in self.__name:
      return {
          "Clean": (self.__data["PPG_Clean"].values)[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Rate": (self.__data["PPG_Rate"].values)[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Peaks": (self.__data["PPG_Peaks"].values)[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
      }
    elif "ECG" in self.__name:
      return {
          "Clean": (self.__data["ECG_Clean"].values)[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Rate": (self.__data["ECG_Rate"].values)[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1],
          "Peaks": (self.__data["ECG_R_Peaks"].values)[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1]
      }
    return self.__data[floor((time_begin - origin) * self.__sampling) : floor((time_end - origin) * self.__sampling) + 1]

"""
Класс одного участника эксперимента
subject_keys = ['singal', 'label', 'subject']
signal_keys = ['chest', 'wrist']
chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
wrist_keys = ['ACC', 'EDA', 'TEMP', 'BVP']
"""
class WesadSubject:
  """
  Считываем данные одного пациента.

  main_path - путь на биомедицинские данные пациентов
  subject_name - номер пациента
  """
  def __init__(self, data_manager, subject_name, print_info=False, metric_conf=None):
    self.data_manager = data_manager
    with open(self.data_manager.get_path(dl.DataType.INTERIM) / (subject_name + '.pkl'), 'rb') as file:
          data = pickle.load(file, encoding='latin1')
    self.__name = subject_name
    self.__data = None

    self.__metrics = {
        "chest" : {
          "ACC" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
          "ECG" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
          "EMG" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
          "EDA" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
          "Temp" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
          "Resp" : ["MAX", "MIN", "STD", "MEAN", "RANGE"]
        },
        "wrist" : {
            "ACC" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
            "EDA" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
            "TEMP" : ["MAX", "MIN", "STD", "MEAN", "RANGE"],
            "BVP" : ["MAX", "MIN", "STD", "MEAN", "RANGE"]
        }
    } if not metric_conf else metric_conf

    self.__restructure_data(data, print_info)

  """
  Возвращает номер пациента в формате SX
  """
  def get_name(self):
    return self.__name

  """
  Устанавливает конфигурацию для метрик
  """
  def set_metrics_conf(self, conf):
    for key in self.__metrics['wrist']:
      self.__metrics['wrist'][key] = conf['wrist'][key]

    for key in self.__metrics['chest']:
      self.__metrics['chest'][key] = conf['chest'][key]

  """
  Возвращает количество метрик
  """
  def get_metric_count(self):
    sm = 0

    for key in self.__metrics['wrist']:
      if key == "ACC":
        sm += len(self.__metrics['wrist'][key]) * 3
      else:
        sm += len(self.__metrics['wrist'][key])

    for key in self.__metrics['chest']:
      if key == "ACC":
        sm += len(self.__metrics['chest'][key]) * 3
      else:
        sm += len(self.__metrics['chest'][key])

    return sm

  """
  Вырезаем кусочек сигнала по данным границам, приводя их к требуемой частоте.
  """
  def __extract_signal_method(self, time_interval, signal_name, signal):
    freq = self.__sampling(signal_name)
    return signal[time_interval[0] * freq : time_interval[1] * freq + 1]

  """
  Реструктуризируем сигнал:
  - Вырезаем только кусочки, которые классифицированы, как baseline и stress
  - Разделяем данные на два класса: baseline (0) и stress (1)
  - Разделяем сигналы между собой. Ex: ACC переводим из (:, 3) в  ACC_i : (:, i). См. структура датасета
  """
  def __restructure_data(self, data, print_info):
    baseline_class_mask = np.where((data['label'] == 1))[0]
    baseline_start, baseline_end = baseline_class_mask[0], baseline_class_mask[-1]

    stress_class_mask = np.where((data['label'] == 2))[0]
    stress_start, stress_end = stress_class_mask[0], stress_class_mask[-1]

    amusement_class_mask = np.where((data['label'] == 3))[0]
    amusement_start, amusement_end = amusement_class_mask[0], amusement_class_mask[-1]

    baseline_data = {"time_interval": (ceil(baseline_start / 700), floor(baseline_end / 700)), 'signals': {}}
    stress_data = {"time_interval": (ceil(stress_start / 700), floor(stress_end / 700)), 'signals': {}}
    amusement_data = {"time_interval": (ceil(amusement_start / 700), floor(amusement_end / 700)), 'signals': {}}

    for device in data['signal']:
      if print_info:
        print('device: ', device)
      for signal_type in data['signal'][device]:
        if print_info:
          print('\ttype: ', signal_type)
        for i in range(data['signal'][device][signal_type].shape[1]):
          signal_name = '_'.join([device, signal_type, str(i)])
          signal = data['signal'][device][signal_type][:, i]

          baseline_data['signals'][signal_name] = self.__extract_signal_method(baseline_data['time_interval'], signal_name, signal)
          stress_data['signals'][signal_name] = self.__extract_signal_method(stress_data['time_interval'], signal_name, signal)
          amusement_data['signals'][signal_name] = self.__extract_signal_method(amusement_data['time_interval'], signal_name, signal)

          if len(baseline_data['signals'][signal_name]) == 0:
            raise ValueError(f"baseline signal {signal_name} has a zero length. The length of the signal is {len(signal)} with time interval {baseline_data['time_interval']}")
          if len(stress_data['signals'][signal_name]) == 0:
            raise ValueError(f"stress signal {signal_name} has a zero length. The length of the signal is {len(signal)} with time interval {stress_data['time_interval']}")
          if len(amusement_data['signals'][signal_name]) == 0:
            raise ValueError(f"amusement signal {signal_name} has a zero length. The length of the signal is {len(signal)} with time interval {amusement_data['time_interval']}")

    self.__data = {'baseline': baseline_data, 'stress': stress_data, 'amusement': amusement_data}

    for signal_name in self.__data['baseline']['signals']:
      self.__data['baseline']['signals'][signal_name] = Signal(signal_name, self.__sampling(signal_name), self.__data['baseline']['signals'][signal_name])
      self.__data['stress']['signals'][signal_name] = Signal(signal_name, self.__sampling(signal_name), self.__data['stress']['signals'][signal_name])
      self.__data['amusement']['signals'][signal_name] = Signal(signal_name, self.__sampling(signal_name), self.__data['amusement']['signals'][signal_name])

      if len(self.__data['baseline']['signals'][signal_name].get_data()) == 0:
        raise ValueError(f"baseline signal {signal_name} has a zero length.")
      if len(self.__data['stress']['signals'][signal_name].get_data()) == 0:
        raise ValueError(f"stress signal {signal_name} has a zero length.")
      if len(self.__data['amusement']['signals'][signal_name].get_data()) == 0:
        raise ValueError(f"amusement signal {signal_name} has a zero length.")

    if print_info:
      print(f"\nИтого количество измерений в каждой классе:\n\tНейтральное состояние длится {self.__data['baseline']['time_interval'][1] - self.__data['baseline']['time_interval'][0] + 1} секунд.\n\tСтрессовое состояние длится {self.__data['stress']['time_interval'][1] - self.__data['stress']['time_interval'][0] + 1} секунд.\n\tСостояние веселья длится {self.__data['amusement']['time_interval'][1] - self.__data['amusement']['time_interval'][0] + 1} секунд.")

  """
  По названию и прибору измерений, записанному в signal_name определяет частоту дискретизации сигнала
  """
  def __sampling(self, signal_name):
    if signal_name.startswith("wrist_ACC"):
      return 32
    if signal_name.startswith("wrist_BVP"):
      return 64
    if signal_name.startswith("wrist_EDA"):
      return 4
    if signal_name.startswith("wrist_TEMP"):
      return 4
    return 700

  """
  Формирование датасета размерности (n x 1 x m), где размерность - (количество всего объектов, количество timesteps, количество метрик)
  """
  def preprocess_data_wots(self, window_size, timestep=45, signal_class="baseline"):
      if not self.__data:
          raise ValueError("There is no data.")
      if signal_class not in ['baseline', 'stress', 'amusement']:
          raise ValueError(f"Incorrect signal_class name.\nThe valid options are 'baseline', 'stress' and 'amusement', but '{signal_class}' was received.")

      start = self.__data[signal_class]['time_interval'][0] + window_size
      end = self.__data[signal_class]['time_interval'][1] + 1
      num_windows = ceil((end - start) / timestep)

      num_metrics = self.get_metric_count()
      dataset = np.zeros((num_windows, 1, num_metrics))

      for i, right_border in enumerate(range(start, end, timestep)):
          metrics = []
          for signal_name in self.__data[signal_class]['signals']:
              divice, name, _ = signal_name.split("_")
              if not self.__metrics[divice][name]:
                continue

              signal = self.__data[signal_class]['signals'][signal_name]


              metrics.extend(self.make_metrics(
                  signal.get_data_segment(right_border - window_size, right_border, self.__data[signal_class]['time_interval'][0]),
                  signal.get_name(),
                  signal.get_freq(),
                  type=signal_class
              ))
          dataset[i, 0, :] = metrics

      return dataset

  """
  Формирование датасета размерности (n x t x m), где размерность - (количество всего объектов, количество timesteps, количество метрик)
  """
  def preprocess_data_wts(self, inner_window_size, outer_window_size, inner_window_timestep=30, outer_window_timestep=45, signal_class="baseline"):
      if not self.__data:
          raise ValueError("There is no data.")
      if inner_window_size > outer_window_size:
          raise ValueError("The size of the inner window is more than the size of the outer window.")
      if signal_class not in ['baseline', 'stress', 'amusement']:
          raise ValueError(f"Incorrect signal_class name.\nThe valid options are 'baseline' and 'stress', but '{signal_class}' was received.")


      start = self.__data[signal_class]['time_interval'][0] + outer_window_size
      end = self.__data[signal_class]['time_interval'][1] + 1
      num_outer_windows = floor((end - start) / outer_window_timestep) + 1
      num_inner_windows = floor((outer_window_size - inner_window_size) / inner_window_timestep) + 1

      num_metrics = self.get_metric_count()
      dataset = np.zeros((num_outer_windows, num_inner_windows, num_metrics))

      for i, outer_right_border in enumerate(range(start, end, outer_window_timestep)):
          for j, inner_right_border in enumerate(np.arange(outer_right_border - outer_window_size + inner_window_size, outer_right_border + 1, inner_window_timestep)):
              metrics = []
              for signal_name in self.__data[signal_class]['signals']:
                  divice, name, _ = signal_name.split("_")
                  if not self.__metrics[divice][name]:
                    continue

                  signal = self.__data[signal_class]['signals'][signal_name]
                  window = signal.get_data_segment(inner_right_border - inner_window_size, inner_right_border, self.__data[signal_class]['time_interval'][0])
                  self.last_time_interval = (inner_right_border - inner_window_size, inner_right_border, self.__data[signal_class]['time_interval'])

                  if len(window) == 0:
                    raise ValueError(f"Given {signal_name} window has a zero length. Here's parameters:\n\ttime_interval: {self.__data[signal_class]['time_interval']},\n\tinner_window_bounders: {(inner_right_border - inner_window_size, inner_right_border)}\n\tlength of the signal: {len(signal.get_data())} with {signal.get_freq()} frequence.")

                  metrics.extend(self.make_metrics(
                      window,
                      signal.get_name(),
                      signal.get_freq(),
                      type=signal_class
                  ))

              dataset[i, j, :] = metrics

      return dataset

  """
  Генерация метрик
  """

  def make_metrics(self, window, signal_name, freq, type='baseline'):
      metrics = []
      names = []

      if "wrist_EDA" in signal_name and self.__metrics['wrist']['EDA'] or "chest_EDA" in signal_name and self.__metrics['chest']['EDA']:
          device = signal_name.split("_")[0]

          for key in self.__metrics[device]['EDA']:
            if key == "EDA_MAX":
              metrics.append(np.max(window["Clean"]))
            elif key == "EDA_MIN":
              metrics.append(np.min(window["Clean"]))
            elif key == "EDA_MEAN":
              metrics.append(np.mean(window["Clean"]))
            elif key == "EDA_STD":
              metrics.append(np.std(window["Clean"]))
            elif key == "EDA_RANGE":
              metrics.append(np.max(window["Clean"]) - np.min(window["Clean"]))
            elif key == "SCR_RANGE":
              metrics.append(np.max(window["Phasic"]) - np.min(window["Phasic"]))
            elif key == "SCL_MEAN":
              metrics.append(np.mean(window["Tonic"]))
            elif key == "SCL_STD":
              metrics.append(np.std(window["Tonic"]))
            elif key == "SCL_MAX":
              metrics.append(np.max(window['Tonic']))
            elif key == "SCL_MIN":
              metrics.append(np.min(window['Tonic']))
            elif key == "SCR_PEAKS_NUMBER":
              metrics.append(np.sum(window['Peaks']))
            elif key == "SCR_MEAN":
              metrics.append(np.mean(window['Phasic']))
            elif key == "SCR_STD":
              metrics.append(np.std(window['Phasic']))
            elif key == "SCR_MIN":
              metrics.append(np.min(window['Phasic']))
            elif key == "SCR_MAX":
              metrics.append(np.max(window['Phasic']))
            elif key == "SCR_PEAKS_MEAN":
              metrics.append(np.mean(window['Clean'][window["Peaks"] == 1]))
            elif key == "SCR_PEAKS_STD":
              metrics.append(np.std(window['Clean'][window["Peaks"] == 1]))
            elif key == "SCR_PEAKS_MIN":
              metrics.append(np.min(window['Clean'][window["Peaks"] == 1]))
            elif key == "SCR_PEAKS_MAX":
              metrics.append(np.max(window['Clean'][window["Peaks"] == 1]))
            elif key == "ALSC":
              metrics.append(np.sum(np.sqrt((window['Phasic'][1:] - window['Phasic'][:-1]) ** 2 + 1)))
            elif key == "INSC":
              metrics.append(np.sum(np.abs(window['Phasic'])))
            elif key == "APSC":
              metrics.append(np.mean(window['Phasic'] ** 2))
            elif key == "RMSC":
              metrics.append(np.sqrt(np.mean(window['Phasic'] ** 2)))
            else:
              raise ValueError(f"There is not {key} metric.")
            names.append(key)

      elif "BVP" in signal_name or "ECG" in signal_name:
          titles = []
          temp_metrics = []
          peaks_enough = (np.sum(window['Peaks']) >= 2)
          divice, signal = ('wrist', "BVP") if "BVP" in signal_name else ('chest', "ECG")


          for key in self.__metrics[divice][signal]:
              if key == "HRV_MEAN":
                  if peaks_enough:
                    titles.append('HRV_MeanNN')
                  else:
                    temp_metrics.append(np.nan)
              elif key == "HRV_STD":
                  if peaks_enough:
                    titles.append('HRV_SDNN')
                  else:
                    temp_metrics.append(0.0) #zero peaks, so zero std
              elif key == "pNN50":
                  if peaks_enough:
                    titles.append('HRV_pNN50')
                  else:
                    temp_metrics.append(0.0)
              elif key == "HR_MEAN":
                  temp_metrics.append(np.mean(window['Rate']))
              elif key == "HR_STD":
                  temp_metrics.append(np.std(window['Rate']))
              else:
                  raise ValueError(f"There is not {key} metric.")
              names.append(key)


          if titles and np.sum(window['Peaks']) >= 2:
            hrv = nk.hrv_time(window['Peaks'], sampling_rate=freq)
            temp_metrics += hrv[titles].iloc[0].tolist()
          elif titles:
            print(len(temp_metrics))

          metrics += temp_metrics

      else:
          if 'wrist' in signal_name:
            for key in self.__metrics['wrist']:
                if key in signal_name:
                    for metric_name in self.__metrics['wrist'][key]:
                        if metric_name == "MAX":
                            metrics.append(np.max(window))
                            names.append(f"{key}_MAX")
                        elif metric_name == "MIN":
                            metrics.append(np.min(window))
                            names.append(f"{key}_MIN")
                        elif metric_name == "MEAN":
                            metrics.append(np.mean(window))
                            names.append(f"{key}_MEAN")
                        elif metric_name == "STD":
                            metrics.append(np.std(window))
                            names.append(f"{key}_STD")
                        else:
                            metrics.append(np.max(window) - np.min(window))
                            names.append(f"{key}_RANGE")
          else:
            for key in self.__metrics['chest']:
              if key in signal_name:
                  for metric_name in self.__metrics['chest'][key]:
                      if metric_name == "MAX":
                          metrics.append(np.max(window))
                          names.append(f"{key}_MAX")
                      elif metric_name == "MIN":
                          metrics.append(np.min(window))
                          names.append(f"{key}_MIN")
                      elif metric_name == "MEAN":
                          metrics.append(np.mean(window))
                          names.append(f"{key}_MEAN")
                      elif metric_name == "STD":
                          metrics.append(np.std(window))
                          names.append(f"{key}_STD")
                      else:
                          metrics.append(np.max(window) - np.min(window))
                          names.append(f"{key}_RANGE")
      return metrics

data_manager = dl.DataManager()
subject = WesadSubject(data_manager, 'S11', print_info=False)

signal_conf = {
        "chest" : {
          "ACC" : [],
          "ECG" : ["HR_MEAN", "HR_STD", "HRV_MEAN", "HRV_STD", "pNN50"],
          "EMG" : [],
          "EDA" : ["EDA_MEAN", "EDA_STD", "SCL_MEAN", "SCL_STD", "SCR_MEAN", "SCR_STD", "EDA_MAX", "EDA_MIN", "SCL_MAX", "SCL_MIN", "SCR_MAX", "SCR_MIN"],
          "Temp" : [],
          "Resp" : []
        },
        "wrist" : {
            "ACC" : [],
            "EDA" : [],
            "TEMP" : [],
            "BVP" : []
        }
    }