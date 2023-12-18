from datetime import timedelta, datetime
from root import root
import numpy as np
import pandas as pd
import torch

from builder.EncoderForecasterBase import EncoderForecasterBase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prj_root = root()


def init_model(in_channels, out_channels, dim):
    encoder = EncoderForecasterBase()
    encoder.init_encoder(input_size=[dim[0], dim[1]],
                         n_layers=5,
                         in_channels=in_channels,
                         out_channels=out_channels)
    encoder.to(device)
    return encoder


def get_features_for_single_models(forecast_start_point, sea_name):
    dates = pd.date_range(datetime.strptime(forecast_start_point, '%Y%m%d') - timedelta(days=104*7),
                          datetime.strptime(forecast_start_point, '%Y%m%d'), freq='7D')
    dates = [d.strftime('%Y%m%d') for d in dates][:104]
    features = []
    for date in dates:
        matrix = np.load(f'{prj_root}/matrices/{sea_name}_sea_osisaf/osi_{sea_name}_{date}.npy')
        features.append(matrix)
    return np.array(features)


def get_real_data(forecast_start_point, sea_name):
    dates = pd.date_range(datetime.strptime(forecast_start_point, '%Y%m%d'),
                          datetime.strptime(forecast_start_point, '%Y%m%d') + timedelta(days=52 * 7), freq='7D')
    dates = [d.strftime('%Y%m%d') for d in dates][:52]
    target = []
    for date in dates:
        matrix = np.load(f'{prj_root}/matrices/{sea_name}_sea_osisaf/osi_{sea_name}_{date}.npy')
        target.append(matrix)
    return np.array(target), dates


def get_baseline_forecast(forecast_start_point, sea_name):
    dates = pd.date_range(datetime.strptime(forecast_start_point, '%Y%m%d'),
                          datetime.strptime(forecast_start_point, '%Y%m%d') + timedelta(days=52 * 7), freq='7D')
    dates = [d.strftime('%Y%m%d') for d in dates][:52]
    baseline = []
    for date in dates:
        matrix = np.load(f'{prj_root}/matrices/{sea_name}_sea_meanyears_prediction/meanyears_{sea_name}_{date}.npy')
        baseline.append(matrix)
    return np.array(baseline)


def get_ensemble_prediction(start_point, sea_name):
    features = torch.Tensor(get_features_for_single_models(start_point, sea_name)).to(device)
    target, target_dates = get_real_data(start_point, sea_name)

    # ЗАМЕНИТЬ НА ОБУЧЕННЫЕ С 1990 ПО 2016
    cnn1 = init_model(104, 52, (target.shape[1], target.shape[2]))
    cnn1.load_state_dict(torch.load(f'{prj_root}/single_models/{sea_name}_104_52_l1(1990-2015).pt'))
    cnn2 = init_model(104, 52, (target.shape[1], target.shape[2]))
    cnn2.load_state_dict(torch.load(f'{prj_root}/single_models/{sea_name}_104_52_ssim(1990-2015).pt'))

    cnn1_prediction = cnn1(features).cpu().detach().numpy()
    cnn2_prediction = cnn2(features).cpu().detach().numpy()
    baseline_prediction = get_baseline_forecast(start_point, sea_name)

    ensemble_features = torch.Tensor(np.vstack((cnn1_prediction, cnn2_prediction, baseline_prediction))).to(device)

    ensemble_cnn = init_model(52*3, 52, (target.shape[1], target.shape[2]))
    ensemble_cnn.load_state_dict(torch.load(f'{prj_root}/ensemble_models/{sea_name}_52_ssim.pt'))

    ensemble_forecast = ensemble_cnn(ensemble_features)
    ensemble_forecast = ensemble_forecast.cpu().detach().numpy()
    ensemble_forecast[ensemble_forecast > 1] = 1

    return ensemble_forecast, cnn1_prediction, cnn2_prediction, baseline_prediction, target, target_dates




