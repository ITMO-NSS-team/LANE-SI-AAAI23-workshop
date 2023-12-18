from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from builder.EdgeExtractor import EdgeExtractor
from builder.METRICS import polyline_dist, ssim, mae
from predict_ensemble import get_ensemble_prediction
from visualization.tricks import rotate, binary
plt.rcParams['image.cmap'] = 'Blues'

sea_name = 'barents'

coastline = rotate(np.load(f'../matrices/{sea_name}_land_mask.npy'))
#mask = rotate(np.load('../matrices/kara_coastline_mask.npy'))
x = np.arange(coastline.shape[1])
y = np.arange(coastline.shape[0])

year = 2022
start_point = f'{year}0101'

ensemble_forecast, cnn1_prediction, cnn2_prediction, baseline_prediction, target, target_dates = get_ensemble_prediction(
    start_point, sea_name)

for i in range(ensemble_forecast.shape[0]):
    if (ensemble_forecast[i] == 0).all():
        ensemble_forecast[i] = ensemble_forecast[i-1]

seas5_prediction = []
for date in target_dates:
    seas5_prediction.append(np.load(f'../matrices/seas5/{sea_name}/seas5_{sea_name}_{date}.npy'))
seas5_prediction = np.array(seas5_prediction)

extractor = EdgeExtractor()

metrics_dict = {'dist': {'seas5': [],
                         'ensemble': []},
                'dist_distr': {'seas5': [],
                               'ensemble': []},
                'mae': {'seas5': [],
                        'ensemble': []},
                'ssim': {'seas5': [],
                         'ensemble': []},
                'dates': []}

img_folder = f'C:/Users/Julia/Pictures/seas_vs_ensemble_edge/{sea_name}/'

for i in range(0, len(target_dates)):
    print(target_dates[i])
    metrics_dict['dates'].append(target_dates[i])
    target_edge = extractor.extract_max_edge(rotate(binary(target[i])), to_length=100)
    ensemble_edge = extractor.extract_max_edge(rotate(binary(ensemble_forecast[i], 0.7)), to_length=100)
    seas5_edge = extractor.extract_max_edge(rotate(binary(seas5_prediction[i], 0.8)), to_length=100)

    ensemble_ssim = ssim(ensemble_forecast[i], target[i])
    seas5_ssim = ssim(seas5_prediction[i], target[i])
    metrics_dict['ssim']['ensemble'].append(ensemble_ssim)
    metrics_dict['ssim']['seas5'].append(seas5_ssim)

    ensemble_mae = mae(ensemble_forecast[i], target[i])
    seas5_mae = mae(seas5_prediction[i], target[i])
    metrics_dict['mae']['ensemble'].append(ensemble_mae)
    metrics_dict['mae']['seas5'].append(seas5_mae)

    ensemble_dist = polyline_dist(target_edge, ensemble_edge)
    seas5_dist = polyline_dist(target_edge, seas5_edge)
    metrics_dict['dist']['ensemble'].append(ensemble_dist)
    metrics_dict['dist']['seas5'].append(seas5_dist)

    ensemble_dist_distr = polyline_dist(target_edge, ensemble_edge, with_interval=True)
    seas5_dist_distr = polyline_dist(target_edge, seas5_edge, with_interval=True)
    metrics_dict['dist_distr']['ensemble'].append(ensemble_dist_distr)
    metrics_dict['dist_distr']['seas5'].append(seas5_dist_distr)

    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    axs = fig.subplot_mosaic([['im1', 'im2', 'im3'],
                              ],
                             gridspec_kw={'width_ratios': [5, 5, 5],
                                          'height_ratios': [6]})

    axs['im1'].set_title(f'{target_dates[i]} - real')
    axs['im1'].imshow(rotate(target[i]), vmax=1, vmin=0)
    axs['im1'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im1'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    if target_edge.shape[1] != 0:
        axs['im1'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
        axs['im1'].plot(target_edge[:, 0], target_edge[:, 1], c='g', linewidth=0.8)

    axs['im2'].set_title(f'Ensemble\ndist={ensemble_dist}\nSSIM={ensemble_ssim}\nMAE={ensemble_mae}')
    axs['im2'].imshow(rotate(ensemble_forecast[i]), vmax=1, vmin=0)
    axs['im2'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im2'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    # axs['im2'].contour(x, y, rotate(ensemble_forecast[i]), [0.8], colors=['r'])
    if target_edge.shape[1] != 0:
        axs['im2'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
        axs['im2'].plot(target_edge[:, 0], target_edge[:, 1], c='g', linewidth=0.8)
    if ensemble_edge.shape[1] != 0:
        axs['im2'].scatter(ensemble_edge[:, 0], ensemble_edge[:, 1], c='r', s=8)
        axs['im2'].plot(ensemble_edge[:, 0], ensemble_edge[:, 1], c='r', linewidth=0.8)

    axs['im3'].set_title(f'SEAS5\ndist={seas5_dist}\nSSIM={seas5_ssim}\nMAE={seas5_mae}')
    axs['im3'].imshow(rotate(seas5_prediction[i]), vmax=1, vmin=0)
    axs['im3'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im3'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    # axs['im3'].contour(x, y, rotate(cnn1_prediction[i]), [0.8], colors=['r'])
    if target_edge.shape[1] != 0:
        axs['im3'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
        axs['im3'].plot(target_edge[:, 0], target_edge[:, 1], c='g', linewidth=0.8)
    if seas5_edge.shape[1] != 0:
        axs['im3'].scatter(seas5_edge[:, 0], seas5_edge[:, 1], c='r', s=8)
        axs['im3'].plot(seas5_edge[:, 0], seas5_edge[:, 1], c='r', linewidth=0.8)
    plt.gcf().text(0.02, 0.95, f'----- prediction ice edge', fontsize=12, c='r')
    plt.gcf().text(0.02, 0.92, f'----- target ice edge', fontsize=12, c='g')
    plt.savefig(f'{img_folder}/{target_dates[i]}.png')
    plt.show()

metrics_dict['dates'] = [datetime.strptime(d, '%Y%m%d') for d in metrics_dict['dates']]

plt.plot(metrics_dict['dates'], metrics_dict['ssim']['seas5'], label='SEAS5')
plt.plot(metrics_dict['dates'], metrics_dict['ssim']['ensemble'], label='Ensemble')
plt.legend()
plt.xticks(rotation=90)
plt.title('SSIM')
plt.tight_layout()
plt.savefig(f'{img_folder}/ssim_{year}.png')
plt.show()

plt.plot(metrics_dict['dates'], metrics_dict['mae']['seas5'], label='SEAS5')
plt.plot(metrics_dict['dates'], metrics_dict['mae']['ensemble'], label='Ensemble')
plt.legend()
plt.xticks(rotation=90)
plt.title('MAE')
plt.tight_layout()
plt.savefig(f'{img_folder}/mae_{year}.png')
plt.show()

plt.plot(metrics_dict['dates'], metrics_dict['dist']['seas5'], label='SEAS5')
plt.plot(metrics_dict['dates'], metrics_dict['dist']['ensemble'], label='Ensemble')
plt.legend()
plt.xticks(rotation=90)
plt.title('dist')
plt.tight_layout()
plt.savefig(f'{img_folder}/dist_{year}.png')
plt.show()

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 10))

dates = pd.Series(pd.to_datetime(metrics_dict['dates']))
dates = dates[dates.dt.month < 6]
ind = len(dates)

axs[0].boxplot(metrics_dict['dist_distr']['seas5'][:ind],  labels=dates, showfliers=False)
axs[0].set_title('SEAS5 edge distance metric')
axs[0].set_ylim(0, 200)
axs[0].set_ylabel('Distance')
axs[0].set_xticks([])


axs[1].boxplot(metrics_dict['dist_distr']['ensemble'][:ind],  labels=dates, showfliers=False)
axs[1].set_title('Ensemble model edge distance metric')
axs[1].set_ylim(0, 200)
axs[1].set_ylabel('Distance')
axs[1].set_xticks([])

axs[2].set_title('Comparison of SEAS5 and ensemble model mean distance metric')
axs[2].plot(dates, metrics_dict['dist']['seas5'][:ind], label='SEAS5')
axs[2].plot(dates, metrics_dict['dist']['ensemble'][:ind], label='Ensemble model')
#axs[2].set_ylim(0, 800)
axs[2].legend()
axs[2].set_ylabel('Distance')
plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=70 )
axs[2].set_xlabel('Forecast date')
plt.tight_layout()
plt.savefig(f'{img_folder}/seas_vs_ensemble_{year}.png')
plt.show()
