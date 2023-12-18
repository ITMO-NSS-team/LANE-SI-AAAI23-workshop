import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from builder.EncoderForecasterBase import EncoderForecasterBase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')

pre_history_size = 104
forecast_size = 52

sea_name = 'barents'

def init_model(in_channels, out_channels, dims):
    encoder = EncoderForecasterBase()
    encoder.init_encoder(input_size=[dims[0], dims[1]],
                         n_layers=5,
                         in_channels=in_channels,
                         out_channels=out_channels)
    encoder.to(device)
    print(encoder)
    return encoder


def init_full_dataset(dates_range):
    dates = []
    dataset = []
    for date in dates_range:
        file =f'osi_{sea_name}_{date}.npy'
        array = np.load(f'matrices/{sea_name}_sea_osisaf/{file}')
        dates.append(date)
        dataset.append(array)
    return dates, np.array(dataset)


def baseline_predict(dates_range):
    baseline_predictions_folder = f'matrices/{sea_name}_sea_meanyears_prediction'
    prediction = []
    for date in dates_range:
        matrix = np.load(f'{baseline_predictions_folder}/meanyears_{sea_name}_{date}.npy')
        prediction.append(matrix)
    return np.array(prediction)


def create_train_dataset_for_ensemble(dates_range):
    features_for_ensemble = []
    target_for_ensemble = []

    all_dates, all_matrices = init_full_dataset(pd.date_range('20090101', '20161231').strftime('%Y%m%d'))

    cnn1 = init_model(pre_history_size, forecast_size, (all_matrices.shape[1], all_matrices.shape[2]))
    cnn1.load_state_dict(torch.load(f'single_models/{sea_name}_104_52_l1.pt'))
    cnn2 = init_model(pre_history_size, forecast_size, (all_matrices.shape[1], all_matrices.shape[2]))
    cnn2.load_state_dict(torch.load(f'single_models/{sea_name}_104_52_ssim.pt'))

    for date in dates_range:
        print(f'Prediction single models for date {date}')
        date_index = all_dates.index(date)
        pre_history_inds = np.arange(date_index-104*7, date_index, 7)
        forecast_size_inds = np.arange(date_index, date_index+52*7, 7)

        sample_features = torch.Tensor(all_matrices[pre_history_inds]).to("cuda")
        sample_target = all_matrices[forecast_size_inds]
        sample_target_dates = np.array(all_dates)[forecast_size_inds]
        # предсказание на горизонт прогноза из каждой даты сэмпла для ансамбля
        cnn1_prediction = cnn1(sample_features).cpu().detach().numpy()
        cnn2_prediction = cnn2(sample_features).cpu().detach().numpy()
        baseline_prediction = baseline_predict(sample_target_dates)

        features_for_ensemble.append(np.vstack((cnn1_prediction, cnn2_prediction, baseline_prediction)))
        target_for_ensemble.append(sample_target)

        '''plt.imshow(cnn1_prediction)
        plt.show()
        plt.imshow(sample_target)
        plt.show()'''

    return np.array(features_for_ensemble), np.array(target_for_ensemble)


def train_ensemble(features, target, epochs, loss_name='l1'):
    batch_size = 10
    ensembling_cnn = init_model(features.shape[1], target.shape[1], (features.shape[2], features.shape[3])).to(device)

    tensor_x, tensor_y = torch.Tensor(features), torch.Tensor(target)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print('Loader created')

    optimizer = optim.Adam(ensembling_cnn.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    losses = []
    start = time.time()
    for epoch in range(epochs):
        loss = 0
        for train_features, test_features in dataloader:
            train_features = train_features.to(device)
            test_features = test_features.to(device)
            optimizer.zero_grad()
            outputs = ensembling_cnn(train_features)
            if loss_name =='l1':
                l1_loss = criterion(outputs, test_features)
                train_loss = l1_loss
            if loss_name == 'ssim':
                ssim_loss = 1 - ssim(outputs, test_features, data_range=1, size_average=True)
                train_loss = ssim_loss
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(dataloader)
        losses.append(loss)
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    end = time.time() - start
    print(f'Runtime seconds: {end}')
    torch.save(ensembling_cnn.state_dict(), f"ensemble_models/{sea_name}_52_{loss_name}.pt")
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()



start_points_dates_range = pd.date_range('20100101', '20151231', freq='15D').strftime('%Y%m%d')
features, target = create_train_dataset_for_ensemble(start_points_dates_range)
train_ensemble(features, target, epochs=1000, loss_name='ssim')