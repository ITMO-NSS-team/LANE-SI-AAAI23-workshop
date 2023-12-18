import os

import numpy as np
import pandas as pd
import torch
import pickle

from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from typing import Callable, Optional


def timestep_from_string(string):
    return string.split('_')[-1][0:8]


class EncoderBased:
    def __init__(self, encoder_model_path: str):
        self.encoder = self._init_pickle_model(encoder_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

    @staticmethod
    def _init_pickle_model(path):
        with open(path, "rb") as fp:
            model = pickle.load(fp)
            return model


class ImagesTsConverter(EncoderBased):
    def __init__(self, encoder_model_path):
        super(ImagesTsConverter, self).__init__(encoder_model_path)
        self.loader = None
        self.timesteps_no_shuffle = []
        self.batch_size = 10

    def init_images_loader(self,
                           matrices_path: str,
                           batch_size: int,
                           timestep_extract: Optional[Callable] = None,
                           shuffle: bool = False):

        self.batch_size = batch_size
        x = []
        timesteps = []

        for file in os.listdir(matrices_path):
            array = np.load(f'{matrices_path}/{file}')
            #array = add_band_dim(array) * 100
            x.append(array)
            if timestep_extract:
                timestep = timestep_extract(file)
                timesteps.append(timestep)

        tensor_x = torch.Tensor(x)
        my_dataset = TensorDataset(tensor_x, tensor_x)
        self.loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)
        if len(timesteps) != 0:
            self.timesteps_no_shuffle = timesteps

    def reconstructe_image(self, one_batch=True):
        full_set_real = []
        full_set_reconstruction = []
        with torch.no_grad():
            for batch_features in self.loader:
                batch_features = batch_features[0]
                real_images = batch_features
                full_set_real.extend(real_images)
                reconstruction = self.encoder(real_images)
                full_set_reconstruction.extend(reconstruction)
                if one_batch:
                    break
        return full_set_real, full_set_reconstruction

    def visualize_reconstruction_quality(self, real_images, reconstruction, with_metric=False):
        for i in range(len(real_images)):
            fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

            plt.sca(axarr[0])
            plt.imshow(real_images[i].numpy()[0])
            plt.title('real')

            plt.sca(axarr[1])
            plt.imshow(reconstruction[i].numpy()[0])
            plt.title('reconstruction')

            if with_metric:
                plt.suptitle(
                    f'MAE = {np.round(np.mean(abs(real_images[i].numpy()[0] - reconstruction[i].numpy()[0])), 4)}'
                )
            plt.show()

    def convert_images_to_dataframe(self):
        multivariate_ts = None
        for batch_features, _ in self.loader:
            batch_features = batch_features.to(self.device)
            mini_ts = self.encoder.encode(batch_features)
            if multivariate_ts is None:
                multivariate_ts = mini_ts.detach().numpy()
            else:
                multivariate_ts = np.vstack((multivariate_ts, mini_ts.detach().numpy()))
        df = pd.DataFrame()
        df['date'] = self.timesteps_no_shuffle
        features_num = multivariate_ts.shape[1]
        for feature in range(features_num):
            df[f'ts_{feature}'] = multivariate_ts[:, feature]
        return df

    def visualize_multimodal_ts(self, df):
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        for column in df.columns:
            if column != 'date':
                plt.plot(df['date'], df[column], label=column)
        plt.legend()
        plt.show()


class TsImageConverter(EncoderBased):
    def __init__(self, encoder_model_path):
        super(TsImageConverter, self).__init__(encoder_model_path)
        self.timeseries_dataset = None

    def init_timeseries_dataset(self, df_path):
        df = pd.read_csv(df_path)
        self.timeseries_dataset = df

    def convert_dataframe_to_images(self, df_path):
        self.init_timeseries_dataset(df_path)
        images_list = []
        rows_num = len(self.timeseries_dataset)
        for row in range(0, rows_num):
            values = self.timeseries_dataset.iloc[row].values
            tensor = self._transform_ts_dataset_to_tensor(values)
            image = self.decode_matrix(tensor)
            images_list.append(image)
        return images_list

    def decode_matrix(self, tensor):
        image = self.encoder.decode(tensor).detach().numpy()[0]
        return image

    def _transform_ts_dataset_to_tensor(self, np_array):
        tensor = torch.from_numpy(np_array).float().to(self.device)
        return tensor
