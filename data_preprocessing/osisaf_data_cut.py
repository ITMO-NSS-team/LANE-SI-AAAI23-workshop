import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from visualization.tricks import rotate

seas_codes = {0: 'Гренландское',
              1: 'Восточно-Сибирское',
              2: 'Чукотское',
              3: 'Бофорта',
              4: 'Лаптевых',
              5: 'Карское',
              6: 'Баренцево'
              }


def save_kara_sea_matrices():
    train_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/train'
    test_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/test'
    mask = np.load('../matrices/seas_mask_82degr.npy')
    plt.imshow(mask)
    plt.show()
    plt.imshow(mask[0:140, 130:250]) #140x120
    plt.show()
    for folder in [train_path, test_path]:
        for file in os.listdir(folder):
            date = file[-12:-4]
            matrix = np.load(f'{folder}/{file}')
            np.save(f'../matrices/kara_sea_osisaf/osi_kara_{date}.npy', matrix[0:140, 130:250])


def save_laptev_sea_matrices():
    train_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/train'
    test_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/test'
    mask = np.load('../matrices/seas_mask_82degr.npy')
    plt.imshow(mask)
    plt.show()
    plt.imshow(mask[50:160, 210:340]) #110x130
    plt.show()
    for folder in [train_path, test_path]:
        for file in os.listdir(folder):
            date = file[-12:-4]
            matrix = np.load(f'{folder}/{file}')[50:160, 210:340]
            np.save(f'../matrices/laptev_sea_osisaf/osi_laptev_{date}.npy', matrix)


def save_barents_sea_matrices():
    train_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/train'
    test_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/test'
    mask = np.load('../matrices/seas_mask_82degr.npy')
    plt.imshow(mask)
    plt.show()
    plt.imshow(mask[20:180, 50:200]) #160x150
    plt.show()
    for folder in [train_path, test_path]:
        for file in os.listdir(folder):
            date = file[-12:-4]
            matrix = np.load(f'{folder}/{file}')[20:180, 50:200]
            np.save(f'../matrices/barents_sea_osisaf/osi_barents_{date}.npy', matrix)


def save_chukchi_sea_matrices():
    train_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/train'
    test_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/test'
    mask = np.load('../matrices/seas_mask_82degr.npy')
    mask[mask!=2]=None
    plt.imshow(mask)
    plt.show()
    plt.imshow(mask[180:265, 260:]) #85x146
    plt.show()
    for folder in [train_path, test_path]:
        for file in os.listdir(folder):
            date = file[-12:-4]
            matrix = np.load(f'{folder}/{file}')[180:265, 260:]
            np.save(f'../matrices/chukchi_sea_osisaf/osi_chukchi_{date}.npy', matrix)


save_chukchi_sea_matrices()


def prepare_mask():
    dates = pd.date_range('19900101', '20091231')
    dates = [date.strftime('%Y%m%d') for date in dates]
    matrix = np.load(f'../matrices/barents_sea_osisaf/osi_barents_19900101.npy')
    matrix_sum = np.zeros((matrix.shape[0], matrix.shape[1]))
    for date in dates:
        matrix = np.load(f'../matrices/barents_sea_osisaf/osi_barents_{date}.npy')
        matrix_sum = matrix_sum+matrix
    matrix_sum[matrix_sum != 0] = 1
    '''matrix_sum[68:, :40] = 1
    matrix_sum[80:, :54] = 1
    matrix_sum[60:, :5] = 1'''
    plt.imshow(matrix_sum)
    plt.show()
    np.save('../matrices/barents_land_mask.npy', matrix_sum)
