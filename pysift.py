import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from scipy import signal
from scipy.ndimage import gaussian_filter

import cv2
import torch
import torch.nn as nn


def get_gradients(W):
    # Gradient computation
    g = np.zeros((3, 1))
    # dW/dh
    g[0, 0] = (W[2, 1, 1] - W[0, 1, 1])/2
    # dW/dw
    g[1, 0] = (W[1, 2, 1] - W[1, 0, 1])/2
    # dW/ds
    g[2, 0] = (W[1, 1, 2] - W[1, 1, 0])/2
    return g


def get_hessian(W):
    # Hessian computation
    H = np.zeros((3, 3))
    H[0, 0] = W[2, 1, 1] + W[0, 1, 1] - 2 * W[1, 1, 1]
    H[1, 1] = W[1, 2, 1] + W[1, 0, 1] - 2 * W[1, 1, 1]
    H[2, 2] = W[1, 1, 2] + W[1, 1, 0] - 2 * W[1, 1, 1]
    H[0, 1] = (W[2, 2, 1] - W[2, 0, 1] - W[0, 2, 1] + W[0, 0, 1])/4
    H[0, 2] = (W[2, 1, 2] - W[2, 1, 0] - W[0, 1, 2] + W[0, 1, 0])/4
    H[1, 2] = (W[1, 2, 2] - W[1, 2, 0] - W[1, 0, 2] + W[1, 0, 0])/4
    H[1, 0] = H[0, 1]
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    return H


def get_image(path):
    image_rgb = cv2.imread(path)
    image = cv2.imread(path, 0)
    image = image / image.max()
    image_rgb = image_rgb / image_rgb.max()
    return image.astype(float)


def get_resized_image(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


def normalize_features(features):
    features = np.minimum(features, 0.2*np.linalg.norm(features))
    norm_features = np.linalg.norm(features)
    if norm_features > 0:
        features = (512*features/norm_features).astype(int)
        features = np.minimum(features, 255)
    return features


class Keypoint:
    def __init__(self, octave, scale, h, w, sigma, x, y):
        self.octave = octave
        self.scale = scale
        self.h = h
        self.w = w
        self.sigma = sigma
        self.x = x
        self.y = y
        self.theta = None
        self.feature = None

    def set_reference(self, theta):
        self.theta = theta

    def set_feature(self, feature):
        self.feature = feature


class Sift:
    ''' This implementation follows:
        https://www.ipol.im/pub/art/2014/82/article_lr.pdf
    '''
    def __init__(self, image_name):
        self.image = get_image(image_name)
        self.nb_octave = int(np.log2(min(self.image.shape[:2])))
        self.nb_octave = 8
        self.nb_scale = 3
        self.sigma_min = 0.8
        self.sigma_in = 0.5
        self.contrast_dog = 0.015
        self.contrast_threshold = (2**(1/self.nb_scale) - 1)/(2**(1/3) - 1) * \
            self.contrast_dog
        self.delta_min = 0.5
        self.lambda_ori = 1.5
        self.c_edge = 10
        self.nb_attempts = 5
        self.factor = 3
        self.n_bins = 36
        self.max_ratio = 0.8
        self.n_hist = 4
        self.n_ori = 8

        # 2.2 Digital Gaussian Scale-Space
        self.construct_gaussian_scale_space()
        # 3.1 Scale-Space Analysis: Difference of Gaussians
        self.compute_difference_of_gaussians()
        self.get_keypoints()
        print('nb_keypoints', len(self.keypoints))
        # 4.2 Keypoint Normalized Descriptor
        self.keypoint_normalized_descriptor()
        print('nb_descriptor_keypoints', len(self.descriptor_keypoints))

    def construct_gaussian_scale_space(self):
        sigma = np.sqrt(self.sigma_min**2 - self.sigma_in**2)
        self.D = []
        sigma = 1 / self.delta_min * \
            np.sqrt(self.sigma_min ** 2 - self.sigma_in ** 2)
        list_of_sigma = [sigma]
        c = self.sigma_min / self.delta_min
        for scale in range(1, self.nb_scale + 3):
            s1 = 2**(2 * (scale - 1) / self.nb_scale)
            s2 = 2**(2 * (scale / self.nb_scale))
            list_of_sigma.append(c * np.sqrt(s2 - s1))

        image_in = self.image
        for o in range(self.nb_octave):
            if o == 0:
                image_in = cv2.resize(image_in, None, fx=2, fy=2,
                                      interpolation=cv2.INTER_LINEAR)
                out = np.zeros((self.nb_scale + 3, image_in.shape[0],
                                image_in.shape[1]))
                image_in = gaussian_filter(image_in, list_of_sigma[0])
            else:
                image_in = cv2.resize(out[-2], None, fx=0.5, fy=0.5,
                                      interpolation=cv2.INTER_LINEAR)
                out = np.zeros((self.nb_scale + 3, image_in.shape[0],
                                image_in.shape[1]))

            out[0] = image_in
            for scale in range(1, self.nb_scale + 3):
                image_in = gaussian_filter(image_in, list_of_sigma[scale])
                out[scale] = image_in

            image_in = out[-1]
            self.D.append(out)

    def compute_difference_of_gaussians(self):
        self.DoG = []
        for o in range(self.nb_octave):
            self.DoG.append(self.D[o][1:] - self.D[o][:-1])

    def get_keypoints(self):
        self.keypoints = []
        for o in range(self.nb_octave):
            cp = 0
            dog = self.DoG[o]
            # 3 Find candidates keypoints
            pooling = nn.MaxPool2d(3, stride=(1, 1), padding=(1, 1))
            out_max = pooling(torch.tensor(dog))
            out_min = -pooling(torch.tensor(-dog))
            for i in range(1, self.nb_scale + 1):
                # Detection of DoG 3D discrete extrema
                idx = self.filter_keypoint(dog[i], out_max, out_min, i)
                cp += idx.shape[0]
                for key in idx:
                    h, w = key
                    if not(1 <= h < dog.shape[1] - 1 and
                           1 <= w < dog.shape[2] - 1):
                        continue
                    # 3.2 Extraction of Candidate Keypoints
                    keypoint = self.find_keypoint(dog, key, i, o)
                    if keypoint is not None:
                        # 4.1 Keypoint Reference Orientation
                        self.get_keypoint_reference_orientation(keypoint)

    def filter_keypoint(self, data, out_max, out_min, i):
        # 3.3 Filtering Unstable Keypoints
        max_scale = np.array(out_max[i-1:i+2].max(dim=0).values)
        min_scale = np.array(out_min[i-1:i+2].min(dim=0).values)
        # Filter keypoints with enough contrast:
        mask_th = np.abs(data) >= 0.8 * self.contrast_threshold
        # Filter keypoints as extremum:
        mask_max = data >= max_scale
        mask_min = data <= min_scale
        mask = np.logical_and(mask_th, np.logical_or(mask_min, mask_max))
        idx = np.where(mask)
        idx = np.stack((idx[0], idx[1]), axis=1)
        return idx

    def find_keypoint(self, dog, key, i, octave):
        # Keypoint position refinement
        keypoint = None
        h, w = key
        s, x, y = i, w, h
        for n in range(self.nb_attempts):
            W = dog[i-1:i+2, h-1:h+2, w-1:w+2]
            g = get_gradients(W)
            H = get_hessian(W)
            alpha = -np.linalg.lstsq(H, g, rcond=None)[0]
            y = (h + alpha[1]) * self.delta_min * 2 ** octave
            x = (w + alpha[2]) * self.delta_min * 2 ** octave
            s += int(np.round(alpha[0]))
            h += int(np.round(alpha[1]))
            w += int(np.round(alpha[2]))
            if max(abs(alpha)) < 0.6:
                break
            if not(0 <= s < self.nb_scale and 1 <= h < dog.shape[1] - 1 and
                    1 <= w < dog.shape[2] - 1):
                return keypoint

        ratio = (self.c_edge + 1) ** 2 / self.c_edge
        extremum = W[1, 1, 1] + 0.5 * g.transpose().dot(alpha)[0][0]
        if abs(extremum) >= self.contrast_threshold:
            hessian_trace = np.trace(H[1:, 1:])
            hessian_det = np.linalg.det(H[1:, 1:])
            if (hessian_trace**2 / (hessian_det + 1e-7) < ratio and
                    hessian_det > 0):
                pixel_dist = self.delta_min * 2**octave
                sigma = (pixel_dist / self.delta_min) * self.sigma_min * \
                    2 ** (s / self.nb_scale)
                keypoint = Keypoint(octave, s, h, w, sigma, x, y)
        return keypoint

    def get_patch_orientation(self, keypoint, pixel_dist):
        size_patch = int(self.factor * self.lambda_ori * keypoint.sigma)

        ym = keypoint.y - size_patch
        yM = keypoint.y + size_patch
        xm = keypoint.x - size_patch
        xM = keypoint.x + size_patch

        hm = int(np.round(ym/pixel_dist))
        hM = int(np.round(yM/pixel_dist))
        wm = int(np.round(xm/pixel_dist))
        wM = int(np.round(xM/pixel_dist))
        return hm, hM, wm, wM

    def get_keypoint_reference_orientation(self, keypoint):
        # 4.1 : Keypoint Reference Orientation
        G = self.D[keypoint.octave][keypoint.scale]

        # A. Orientation histogram accumulation
        pixel_dist = self.delta_min * 2**keypoint.octave
        histogram = np.zeros(self.n_bins)
        hm, hM, wm, wM = self.get_patch_orientation(keypoint, pixel_dist)

        if not(0 < hm and hM < G.shape[0] - 1 and
               0 < wm and wM < G.shape[1] - 1):
            return

        k = 2 * (self.lambda_ori * keypoint.sigma)**2
        for h in range(hm, hM + 1):
            for w in range(wm, wM + 1):
                dh = (G[h - 1, w] - G[h + 1, w])/2
                dw = (G[h, w + 1] - G[h, w - 1])/2
                c_ori = np.exp(-((keypoint.y - h * pixel_dist)**2 +
                                 (keypoint.x - w * pixel_dist)**2) / k)
                c_ori *= np.sqrt(dh * dh + dw * dw)
                bin_index = int(np.round(self.n_bins * np.arctan2(dh, dw) / 2 / np.pi))
                histogram[bin_index % self.n_bins] += c_ori

        # B. Smoothing the histogram
        for i in range(6):
            histogram = signal.convolve(histogram, np.array([1., 1., 1.])/3)

        # C. Extraction of reference orientation
        global_maximum = max(histogram)
        for k in range(self.n_bins):
            hk = histogram[k]
            k_minus = (k - 1) % self.n_bins
            k_plus = (k + 1) % self.n_bins
            hk_minus = histogram[k_minus]
            hk_plus = histogram[k_plus]
            if hk >= self.max_ratio * global_maximum and hk > max(hk_minus, hk_plus):
                hk_ratio = (hk_minus - hk_plus) / (hk_minus - 2 * hk + hk_plus)
                theta_k = 2 * np.pi * k / self.n_bins
                theta = np.pi / self.n_bins * hk_ratio
                theta_ref = theta_k + theta
                keypoint.set_reference(theta_ref)
                self.keypoints.append(keypoint)

    def get_xy_descriptor(self, keypoint, h, w):
        pixel_dist = self.delta_min * 2**keypoint.octave
        y = (h*pixel_dist - keypoint.y) * np.cos(keypoint.theta) + \
            (w*pixel_dist - keypoint.x) * np.sin(keypoint.theta)
        y /= keypoint.sigma
        x = -(h*pixel_dist - keypoint.y) * np.sin(keypoint.theta) + \
            (w*pixel_dist - keypoint.x) * np.cos(keypoint.theta)
        x /= keypoint.sigma
        return int(x), int(y)

    def get_patch_descriptor(self, keypoint, pixel_dist):
        size_patch = int(round(np.sqrt(2) * 6 * keypoint.sigma *
                               (self.n_hist + 1)/self.n_hist))
        hm = int(np.round((keypoint.y - size_patch)/pixel_dist))
        hM = int(np.round((keypoint.y + size_patch)/pixel_dist))
        wm = int(np.round((keypoint.x - size_patch)/pixel_dist))
        wM = int(np.round((keypoint.y + size_patch)/pixel_dist))
        return hm, hM, wm, wM

    def get_features(self, x, y, theta_h_w, c_descr):
        features = np.zeros(self.n_hist * self.n_hist * self.n_ori)

        for i in range(self.n_hist):
            xi = (i+1 - (1+self.n_hist)/2)*2*6/self.n_hist
            for j in range(self.n_hist):
                yj = (j+1 - (1+self.n_hist)/2)*2*6/self.n_hist
                if (abs(xi - x) <= 2 * 6/self.n_hist and abs(yj - y) <= 2 * 6/self.n_hist):
                    for k in range(self.n_ori):
                        theta = 2 * np.pi * k / self.n_ori
                        if abs((theta - theta_h_w) % (2 * np.pi)) <= 2 * np.pi / self.n_ori:
                            x_contrib = 1 - self.n_hist*abs(xi - x)/2/6
                            y_contrib = 1 - self.n_hist*abs(yj - y)/2/6
                            theta_contrib = 1 - self.n_ori*abs((theta - theta_h_w) % (2 * np.pi))/2/np.pi
                            idx = i*self.n_hist*self.n_ori + j*self.n_ori + k
                            features[idx] += x_contrib * y_contrib * \
                                theta_contrib * c_descr
        return features

    def keypoint_normalized_descriptor(self):
        self.descriptor_keypoints = []
        for keypoint in self.keypoints:
            G = self.D[keypoint.octave][keypoint.scale]

            # keypoint_normalized_descriptor
            pixel_dist = self.delta_min * 2**keypoint.octave
            hm, hM, wm, wM = self.get_patch_descriptor(keypoint, pixel_dist)

            if not(0 < hm and hM < G.shape[0] - 1 and
                   0 < wm and wM < G.shape[1] - 1):
                continue

            c = 2 * (self.lambda_ori * keypoint.sigma)**2

            # The SIFT feature vector.
            features = np.zeros(self.n_hist * self.n_hist * self.n_ori)

            for h in range(hm, hM + 1):
                for w in range(wm, wM + 1):
                    x, y = self.get_xy_descriptor(keypoint, h, w)
                    if max(np.abs([x, y])) < 6 * (self.n_hist + 1)/self.n_hist:
                        dh = (G[h - 1, w] - G[h + 1, w])/2
                        dw = (G[h, w + 1] - G[h, w - 1])/2

                        theta_h_w = (np.arctan2(dh, dw) - keypoint.theta)
                        theta_h_w %= 2 * np.pi
                        c_descr = np.exp(-((keypoint.y - h * pixel_dist)**2 +
                                           (keypoint.x - w * pixel_dist)**2)/c)
                        c_descr *= np.sqrt(dh * dh + dw * dw)

                        features += self.get_features(x, y, theta_h_w, c_descr)

            features = normalize_features(features)
            keypoint.set_feature(features)
            self.descriptor_keypoints.append(keypoint)


def get_matches(sift_A, sift_B):
    matches = []
    descriptor_keypoints_A = sift_A.descriptor_keypoints
    descriptor_keypoints_B = sift_B.descriptor_keypoints
    for descriptor_keypoint_A in descriptor_keypoints_A:
        feature_A = descriptor_keypoint_A.feature

        candidate = None
        min_dist = float('Inf')
        nearest_min = float('Inf')
        for descriptor_keypoint_B in descriptor_keypoints_B:
            feature_B = descriptor_keypoint_B.feature
            d = np.linalg.norm(feature_A - feature_B)
            if d < min_dist:
                min_dist = d
                candidate = descriptor_keypoint_B
            elif d < nearest_min:
                nearest_min = d
        if min_dist < 0.6 * nearest_min:
            matches.append((descriptor_keypoint_A, candidate))

    return matches


def plot_matches(sift_A, sift_B, matches):
    img_A = sift_A.image
    img_B = sift_B.image

    coords_A = [(int(match[0].w * 0.5 * 2**match[0].octave),
                 int(match[0].h * 0.5 * 2**match[0].octave))
                for match in matches]
    coords_A_y = [coord[1] for coord in coords_A]
    coords_A_x = [coord[0] for coord in coords_A]
    coords_A_xy = [(x, y) for x, y in zip(coords_A_x, coords_A_y)]
    coords_B = [(int(match[1].w * 0.5 * 2**match[1].octave),
                 int(match[1].h * 0.5 * 2**match[1].octave))
                for match in matches]
    coords_B_y = [coord[1] for coord in coords_B]
    coords_B_x = [coord[0] for coord in coords_B]
    coords_B_xy = [(x, y) for x, y in zip(coords_B_x, coords_B_y)]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(img_A, cmap='Greys_r')
    ax2.imshow(img_B, cmap='Greys_r')

    ax1.scatter(coords_A_x, coords_A_y)
    ax2.scatter(coords_B_x, coords_B_y)

    for p1, p2 in zip(coords_A_xy, coords_B_xy):
        con = ConnectionPatch(xyA=p2, xyB=p1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="red")
        _ = ax2.add_artist(con)

    plt.show()


def main():
    img_name_A = './image/church.jpg'
    img_name_B = './image/church_transform.jpg'
    sift_A = Sift(img_name_A)
    sift_B = Sift(img_name_B)
    # 5 Matching
    matches = get_matches(sift_A, sift_B)
    print('nb_matches', len(matches))
    plot_matches(sift_A, sift_B, matches)


if __name__ == '__main__':
    main()
