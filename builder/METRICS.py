import numpy as np
from skimage.metrics import structural_similarity
import cv2

def mae(prediction, real):
    return round(float(np.mean(abs(np.array(prediction) - np.array(real)))), 3)


def ssim(prediction, real):
    return round(structural_similarity(prediction, real, data_range=1), 3)


def accuracy(prediction, real):
    f = 125 * 125
    diff = prediction - real
    unique, counts = np.unique(diff, return_counts=True)
    right_calculated = dict(zip(unique, counts))[0]
    return round(right_calculated / f, 3)


def accuracy_without_mask(pred, real, mask):
    pred = pred[mask == 1]
    real = real[mask == 1]
    f = pred.shape[0]
    diff = pred - real
    unique, counts = np.unique(diff, return_counts=True)
    right_calculated = dict(zip(unique, counts))[0]
    return round(right_calculated / f, 3)


def correlation(prediction, real):
    return np.corrcoef(prediction, real)[0, 1]


def point_to_polyline_dist(p, polyline):
    cnt = np.concatenate((polyline, polyline[::-1]))
    return np.abs(cv2.pointPolygonTest(cnt, (int(p[0]), int(p[1])), True))


def polyline_dist(polyline1, polyline2, with_interval=False):
    polyline1 = np.array(polyline1).astype('float32')
    polyline2 = np.array(polyline2).astype('float32')
    if polyline1.shape[1] == 0 or polyline2.shape[1] == 0:
        dist_list = []
    else:
        dist_list = [point_to_polyline_dist(p, polyline2)**2 for p in polyline1]
    if with_interval:
        return dist_list
    return np.round(np.mean(dist_list))
