import os
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

nc_path = 'D:/ice_sources/seas5/6_month_predictions/'
matrices_path = 'D:/ice_sources/seas5/6_month_predictions/full_matrices/'


def seas5_nc_to_npy():
    for file in os.listdir(nc_path):
        if 'gr' in file:
            ds = nc.Dataset(f'{nc_path}/{file}')
            timesteps = np.array(ds.variables['time']).tolist()
            timesteps = [(datetime(1900, 1, 1) + timedelta(hours=d-24)).strftime('%Y%m%d') for d in timesteps]
            for i, time in enumerate(timesteps):
                print(time)
                var_matrix = np.array(ds.variables['siconc'][i])
                var_matrix[var_matrix == -32767] = np.nan
                var_matrix = np.mean(var_matrix, axis=0)
                #plt.imshow(var_matrix)
                #plt.show()
                np.save(f'{matrices_path}/seas_{time}.npy', np.float32(var_matrix))
            ds.close()


def save_kara_seas5():
    mask = np.load('../matrices/kara_land_mask.npy')
    for file in os.listdir(matrices_path):
        date = file[-12:-4]
        matrix = np.load(f'{matrices_path}/{file}')[0:140, 130:250]
        matrix[mask==0]=0
        plt.imshow(matrix)
        plt.show()
        np.save(f'../matrices/seas5/kara/seas5_kara_{date}.npy', matrix)


def save_laptev_seas5():
    mask = np.load('../matrices/laptev_land_mask.npy')
    for file in os.listdir(matrices_path):
        date = file[-12:-4]
        matrix = np.load(f'{matrices_path}/{file}')[50:160, 210:340]
        matrix[mask==0]=0
        plt.imshow(matrix)
        plt.show()
        np.save(f'../matrices/seas5/laptev/seas5_laptev_{date}.npy', matrix)


def save_barents_seas5():
    mask = np.load('../matrices/barents_land_mask.npy')
    for file in os.listdir(matrices_path):
        date = file[-12:-4]
        matrix = np.load(f'{matrices_path}/{file}')[20:180, 50:200]
        matrix[mask==0]=0
        plt.imshow(matrix)
        plt.show()
        np.save(f'../matrices/seas5/barents/seas5_barents_{date}.npy', matrix)
