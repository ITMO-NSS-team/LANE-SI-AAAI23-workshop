import numpy as np
import pandas as pd


def save_meanyears_prediction(years, sea_name):
    for year in years:
        prediction_dates = pd.date_range(f'{year}0101', f'{year}1231', freq='D')
        prediction_dates = [d.strftime('%Y%m%d') for d in prediction_dates]
        means_list = []
        prev_years = np.arange(year-5, year)
        for prev_year in prev_years:
            years_means_list = []
            year_dates = pd.date_range(f'{prev_year}0101', f'{prev_year}1231', freq='D')
            year_dates = [d.strftime('%Y%m%d') for d in year_dates]
            for date in year_dates:
                matrix = np.load(f'../matrices/{sea_name}_sea_osisaf/osi_{sea_name}_{date}.npy')
                years_means_list.append(matrix)
            if len(years_means_list) > 365:
                years_means_list = years_means_list[:-1]
            years_means_list = np.array(years_means_list)
            means_list.append(years_means_list)
        means_list = np.array(means_list)
        means_list = np.mean(means_list, axis=0)
        if len(prediction_dates)>means_list.shape[0]:
            a = np.expand_dims(means_list[-1, :, :], axis=0)
            means_list = np.append(means_list, a, axis=0)
        for i in range(len(prediction_dates)):
            np.save(f'../matrices/{sea_name}_sea_meanyears_prediction/meanyears_{sea_name}_{prediction_dates[i]}.npy', means_list[i])


save_meanyears_prediction(range(2020, 2024), 'laptev')
