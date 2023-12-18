import math
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from skimage.filters.rank import maximum
from skimage.morphology import disk


def area(vs):
    if len(vs)==0:
        return 0
    y = vs[:, 1]
    x = vs[:, 0]
    area=np.abs(0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y)))
    return area


class EdgeExtractor:
    def __init__(self, coastline_mask: Optional = None):
        super(EdgeExtractor, self).__init__()
        self.image_shape = None
        self.smooth_kernel = 2
        self.min_edge_area_percent = 0.03
        self.coastline_mask = coastline_mask

    def _add_borders(self, image):
        image[:, image.shape[1] - 1] = 0
        image[:, 0] = 0
        image[image.shape[0] - 1, :] = 0
        image[0, :] = 0
        return image

    def _remove_borders(self, edge, image):
        edge = np.delete(edge, np.where(edge == image.shape[0] - 1)[0], 0)
        edge = np.delete(edge, np.where(edge == image.shape[1] - 1)[0], 0)
        edge = np.delete(edge, np.where(edge == 0)[0], 0)
        return edge

    def _remove_coastline(self, edge):
        mask_shape = self.coastline_mask.shape
        sm_coastline_mask = maximum(self.coastline_mask, disk(self.smooth_kernel))
        x = np.arange(0, mask_shape[1])
        y = np.arange(0, mask_shape[0])
        plt.imshow(sm_coastline_mask)
        cs = plt.contour(x, y, sm_coastline_mask, [0], colors=['red'])
        plt.close()
        coastline_edges = cs.allsegs[0]
        coastline_edge = []
        for e in coastline_edges:
            if e.shape[0] > 50:
                coastline_edge.extend(list(np.unique(e, axis=0)))
        new_segment = []
        for i in edge:
            correct = True
            for j in coastline_edge:
                if correct:
                    if euclidean(i, j) < 5:
                        correct = False
            if correct:
                new_segment.append(i)
        return np.array(new_segment)

    def filter_to_length(self, segment, n_points):
        if len(segment) == 0:
            return np.array([[],])
        if len(segment) > n_points:
            dif = len(segment)-n_points
            ind_to_del = np.linspace(0, len(segment), dif, endpoint=False).astype(int)
            segment = np.delete(segment, ind_to_del, 0)
        if len(segment) < n_points:
            dif = n_points - len(segment)
            inds = np.linspace(1, len(segment), dif, endpoint=False).astype(int)
            values = np.mean(np.array([segment[inds], segment[inds-1]]), axis=0).astype(int)
            segment = np.insert(segment, inds, values, axis=0)
        return np.array(segment)

    def extract_max_edge(self, image, to_length=None):
        self.image_shape = image.shape
        sm_array = self._add_borders(maximum(image, disk(self.smooth_kernel)))
        image_shape = image.shape
        x = np.arange(0, image_shape[1])
        y = np.arange(0, image_shape[0])
        plt.imshow(sm_array)
        cs = plt.contour(x, y, sm_array, [0])
        plt.close()
        p = cs.allsegs[0]

        segments_area = [area(segment) for segment in p]
        if len(segments_area) == 0:
            return np.array([[],])
        index_of_max_segment = segments_area.index(max(segments_area))
        #max_segment_raw = np.unique(p[index_of_max_segment], axis=0)
        max_segment_raw = p[index_of_max_segment]
        #max_segment = self._remove_borders(np.array(max_segment_raw), sm_array)
        max_segment = max_segment_raw
        if self.coastline_mask is not None:
            max_segment = self._remove_coastline(max_segment)
        if to_length is not None:
            max_segment = self.filter_to_length(max_segment, to_length)
        if area(max_segment_raw) < self.min_edge_area_percent*self.image_shape[0]*self.image_shape[1]:
            return np.array([[],])
        return max_segment

    def extract_edges(self, image):
        sm_array = self._add_borders(maximum(image, disk(self.smooth_kernel)))
        image_shape = image.shape
        x = np.arange(0, image_shape[0])
        y = np.arange(0, image_shape[1])
        plt.imshow(sm_array)
        cs = plt.contour(x, y, sm_array, [0])
        plt.close()
        p = cs.allsegs[0]
        return p
