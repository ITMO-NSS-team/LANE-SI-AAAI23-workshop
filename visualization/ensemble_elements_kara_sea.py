from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from builder.EdgeExtractor import EdgeExtractor
from builder.METRICS import polyline_dist, ssim, mae
from predict_ensemble import get_ensemble_prediction
from visualization.tricks import rotate, binary

plt.rcParams['image.cmap'] = 'Blues'

coastline = rotate(np.load('../matrices/kara_land_mask.npy'))
'''mask = deepcopy(coastline)
mask[:44, :] = 1
mask[40:75, :51] = 1
mask[75:86, :15] = 1
np.save('../matrices/kara_coastline_mask.npy', rotate(mask))'''
mask = rotate(np.load('../matrices/kara_coastline_mask.npy'))
x = np.arange(coastline.shape[1])
y = np.arange(coastline.shape[0])

start_point = '20210101'
ensemble_forecast, cnn1_prediction, cnn2_prediction, baseline_prediction, target, target_dates = get_ensemble_prediction(
    start_point, 'kara')

extractor = EdgeExtractor(mask)

metrics_dict = {'dist': {'meanyears': [],
                         'cnn1': [],
                         'cnn2': [],
                         'ensemble': []},
                'mae': {'meanyears': [],
                        'cnn1': [],
                        'cnn2': [],
                        'ensemble': []},
                'ssim': {'meanyears': [],
                         'cnn1': [],
                         'cnn2': [],
                         'ensemble': []},
                'dates':[]}

img_folder = 'C:/Users/Julia/Pictures/ensemble_per_seas/kara_sea'

for i in range(0, len(target_dates)):
    np.save(f'../matrices/kara_sea_ensemble_prediction/ens_{target_dates[i]}.npy', ensemble_forecast[i])
    print(target_dates[i])
    metrics_dict['dates'].append(target_dates[i])
    target_edge = extractor.extract_max_edge(rotate(binary(target[i])), to_length=100)
    meanyears_edge = extractor.extract_max_edge(rotate(binary(baseline_prediction[i])), to_length=100)
    cnn1_edge = extractor.extract_max_edge(rotate(binary(cnn1_prediction[i])), to_length=100)
    cnn2_edge = extractor.extract_max_edge(rotate(binary(cnn2_prediction[i])), to_length=100)
    ensemble_edge = extractor.extract_max_edge(rotate(binary(ensemble_forecast[i])), to_length=100)

    meanyears_ssim = ssim(baseline_prediction[i], target[i])
    cnn1_ssim = ssim(cnn1_prediction[i], target[i])
    cnn2_ssim = ssim(cnn2_prediction[i], target[i])
    ensemble_ssim = ssim(ensemble_forecast[i], target[i])
    metrics_dict['ssim']['meanyears'].append(meanyears_ssim)
    metrics_dict['ssim']['cnn1'].append(cnn1_ssim)
    metrics_dict['ssim']['cnn2'].append(cnn2_ssim)
    metrics_dict['ssim']['ensemble'].append(ensemble_ssim)

    meanyears_mae = mae(baseline_prediction[i], target[i])
    cnn1_mae = mae(cnn1_prediction[i], target[i])
    cnn2_mae = mae(cnn2_prediction[i], target[i])
    ensemble_mae = mae(ensemble_forecast[i], target[i])
    metrics_dict['mae']['meanyears'].append(meanyears_mae)
    metrics_dict['mae']['cnn1'].append(cnn1_mae)
    metrics_dict['mae']['cnn2'].append(cnn2_mae)
    metrics_dict['mae']['ensemble'].append(ensemble_mae)

    meanyears_dist = polyline_dist(target_edge, meanyears_edge)
    cnn1_dist = polyline_dist(target_edge, cnn1_edge)
    cnn2_dist = polyline_dist(target_edge, cnn2_edge)
    ensemble_dist = polyline_dist(target_edge, ensemble_edge)
    metrics_dict['dist']['meanyears'].append(meanyears_dist)
    metrics_dict['dist']['cnn1'].append(cnn1_dist)
    metrics_dict['dist']['cnn2'].append(cnn2_dist)
    metrics_dict['dist']['ensemble'].append(ensemble_dist)


    fig = plt.figure(constrained_layout=True, figsize=(11, 9))
    axs = fig.subplot_mosaic([['im1', 'im2', 'im3'],
                              ['im1', 'im4', 'im5']
                              ],
                             gridspec_kw={'width_ratios': [5, 5, 5],
                                          'height_ratios': [9, 9]})

    axs['im1'].set_title(f'{target_dates[i]} - real', c='r')
    axs['im1'].imshow(rotate(target[i]), vmax=1, vmin=0)
    axs['im1'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im1'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    if target_edge.shape[1]!=0:
        axs['im1'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)

    axs['im2'].set_title(f'Meanyears\ndist={meanyears_dist}\nSSIM={meanyears_ssim}\nMAE={meanyears_mae}')
    axs['im2'].imshow(rotate(baseline_prediction[i]), vmax=1, vmin=0)
    axs['im2'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im2'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    # axs['im2'].contour(x, y, rotate(baseline_prediction[i]), [0.8], colors=['r'])
    if target_edge.shape[1] != 0:
        axs['im2'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
    if meanyears_edge.shape[1] != 0:
        axs['im2'].scatter(meanyears_edge[:, 0], meanyears_edge[:, 1], c='r', s=8)

    axs['im3'].set_title(f'L1 CNN\ndist={cnn1_dist}\nSSIM={cnn1_ssim}\nMAE={cnn1_mae}')
    axs['im3'].imshow(rotate(cnn1_prediction[i]), vmax=1, vmin=0)
    axs['im3'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im3'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    # axs['im3'].contour(x, y, rotate(cnn1_prediction[i]), [0.8], colors=['r'])
    if target_edge.shape[1] != 0:
        axs['im3'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
    if cnn1_edge.shape[1] != 0:
        axs['im3'].scatter(cnn1_edge[:, 0], cnn1_edge[:, 1], c='r', s=8)

    axs['im4'].set_title(f'SSIM CNN\ndist={cnn2_dist}\nSSIM={cnn2_ssim}\nMAE={cnn2_mae}')
    axs['im4'].imshow(rotate(cnn2_prediction[i]), vmax=1, vmin=0)
    axs['im4'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im4'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    # axs['im4'].contour(x, y, rotate(cnn2_prediction[i]), [0.8], colors=['r'])
    if target_edge.shape[1] != 0:
        axs['im4'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
    if cnn2_edge.shape[1] != 0:
        axs['im4'].scatter(cnn2_edge[:, 0], cnn2_edge[:, 1], c='r', s=8)

    axs['im5'].set_title(f'Ensemble\ndist={ensemble_dist}\nSSIM={ensemble_ssim}\nMAE={ensemble_mae}')
    axs['im5'].imshow(rotate(ensemble_forecast[i]), vmax=1, vmin=0)
    axs['im5'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    # axs['im5'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    # axs['im5'].contour(x, y, rotate(ensemble_forecast[i]), [0.8], colors=['r'])
    if target_edge.shape[1] != 0:
        axs['im5'].scatter(target_edge[:, 0], target_edge[:, 1], c='lime', s=8)
    if ensemble_edge.shape[1] != 0:
        axs['im5'].scatter(ensemble_edge[:, 0], ensemble_edge[:, 1], c='r', s=8)
    plt.savefig(f'{img_folder}/ensemble_elements/{target_dates[i]}.png')
    plt.show()

metrics_dict['dates'] = [datetime.strptime(d, '%Y%m%d') for d in metrics_dict['dates']]

plt.plot(metrics_dict['dates'], metrics_dict['ssim']['meanyears'], label='meanyears')
plt.plot(metrics_dict['dates'], metrics_dict['ssim']['cnn1'], label='cnn1 - l1')
plt.plot(metrics_dict['dates'], metrics_dict['ssim']['cnn2'], label='cnn2 - ssim')
plt.plot(metrics_dict['dates'], metrics_dict['ssim']['ensemble'], label='ensemble')
plt.legend()
plt.xticks(rotation=90)
plt.title('SSIM')
plt.tight_layout()
plt.savefig(f'{img_folder}/ssim_2022.png')
plt.show()


plt.plot(metrics_dict['dates'], metrics_dict['mae']['meanyears'], label='meanyears')
plt.plot(metrics_dict['dates'], metrics_dict['mae']['cnn1'], label='cnn1 - l1')
plt.plot(metrics_dict['dates'], metrics_dict['mae']['cnn2'], label='cnn2 - ssim')
plt.plot(metrics_dict['dates'], metrics_dict['mae']['ensemble'], label='ensemble')
plt.legend()
plt.xticks(rotation=90)
plt.title('MAE')
plt.tight_layout()
plt.savefig(f'{img_folder}/mae_2022.png')
plt.show()


plt.plot(metrics_dict['dates'], metrics_dict['dist']['meanyears'], label='meanyears')
plt.plot(metrics_dict['dates'], metrics_dict['dist']['cnn1'], label='cnn1 - l1')
plt.plot(metrics_dict['dates'], metrics_dict['dist']['cnn2'], label='cnn2 - ssim')
plt.plot(metrics_dict['dates'], metrics_dict['dist']['ensemble'], label='ensemble')
plt.legend()
plt.xticks(rotation=90)
plt.title('dist')
plt.tight_layout()
plt.savefig(f'{img_folder}/dist_2022.png')
plt.show()

