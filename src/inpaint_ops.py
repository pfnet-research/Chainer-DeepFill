import random
import math

import chainer
from chainer import cuda
from chainer import links as L
from chainer import functions as F
from chainer.link_hooks.spectral_normalization import SpectralNormalization
import cv2
import numpy as np


class GenConv(chainer.Chain):
    def __init__(self, ch0, ch1, ksize, stride=1, rate=1, padding="SAME", activation="elu", gated=False):
        super(GenConv, self).__init__()
        self.activation = eval("F." + activation) if activation else None
        self.gated = gated
        initializer = chainer.initializers.GlorotUniform()

        if padding == "SAME":
            pad = ((ksize - 1) * rate + 1 - stride + 1) // 2
        else:
            assert False
        with self.init_scope():
            if gated:
                self.conv = L.Convolution2D(ch0, ch1 * 2, ksize, stride, pad, dilate=rate, initialW=initializer)
            else:
                self.conv = L.Convolution2D(ch0, ch1, ksize, stride, pad, dilate=rate, initialW=initializer)

    def __call__(self, x):
        h = self.conv(x)
        if self.gated:
            h, mask = F.split_axis(h, 2, axis=1)
        if self.activation:
            h = self.activation(h)
        if self.gated:
            h = h * F.sigmoid(mask)
        return h


class GenDeconv(chainer.Chain):
    def __init__(self, ch0, ch1, padding="SAME", gated=False):
        super(GenDeconv, self).__init__()

        with self.init_scope():
            self.genconv = GenConv(ch0, ch1, 3, 1, padding=padding, gated=gated)

    def __call__(self, x):
        h_size, w_size = x.shape[2:]
        h = F.unpooling_2d(x, ksize=2, outsize=(h_size * 2, w_size * 2))
        h = self.genconv(h)
        return h


class DisConv(chainer.Chain):
    def __init__(self, ch0, ch1, ksize=5, stride=2, sn=False):
        super(DisConv, self).__init__()

        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        pad = (ksize - stride + 1) // 2
        with self.init_scope():
            if sn:
                self.conv = L.Convolution2D(ch0, ch1, ksize, stride, pad, initialW=initializer).add_hook(
                    SpectralNormalization())
            else:
                self.conv = L.Convolution2D(ch0, ch1, ksize, stride, pad, initialW=initializer)

    def __call__(self, x):
        h = F.leaky_relu(self.conv(x))
        return h


def random_bbox(config):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - config.VERTICAL_MARGIN - config.HEIGHT
    maxl = img_width - config.HORIZONTAL_MARGIN - config.WIDTH
    t = int(random.uniform(config.VERTICAL_MARGIN, maxt))
    l = int(random.uniform(config.HORIZONTAL_MARGIN, maxl))
    h = config.HEIGHT
    w = config.WIDTH
    return t, l, h, w


def bbox2mask(bbox, batchsize, config, xp):
    """Generate mask tensor from bbox.
    Args:
        bbox: configuration tuple, (top, left, height, width)
        batchsize: batchsize
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        xp: numpy or cupy
    """

    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((batchsize, 1, height, width), np.float32)
        h = np.random.randint(delta_h // 2 + 1)
        w = np.random.randint(delta_w // 2 + 1)
        mask[:, :, bbox[0] + h:bbox[0] + bbox[2] - h,
        bbox[1] + w:bbox[1] + bbox[3] - w] = 1.
        return mask

    img_shape = config.IMG_SHAPES
    height = img_shape[0]
    width = img_shape[1]
    mask = npmask(bbox, height, width,
                  config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH)
    return xp.array(mask)


def free_form_mask(xp, batchsize, size=(256, 256), maxVertex=20, minLength=50,
                   maxLength=200, minBrushWidth=10, maxBrushWidth=40, maxAngle=20):
    """Generate free-form mask tensor
    """
    imageHeight, imageWidth = size
    mask = np.zeros((imageHeight, imageWidth), dtype="float32")
    numVertex = int(random.uniform(2, maxVertex))
    startX = int(random.uniform(0, imageWidth - 1))
    startY = int(random.uniform(0, imageHeight - 1))
    for i in range(numVertex):
        angle = random.uniform(0, maxAngle)
        if i % 2 == 0:
            angle = 180 - angle
        length = random.uniform(minLength, maxLength)
        brushWidth = int(random.uniform(minBrushWidth, maxBrushWidth))
        endX = startX + int(length * np.sin(np.deg2rad(angle)))
        endY = startY + int(length * np.cos(np.deg2rad(angle)))
        cv2.line(mask, (startX, startY), (endX, endY), 255, brushWidth)
        startX = endX
        startY = endY
    mask = mask.reshape(1, 1, imageHeight, imageWidth)
    mask = np.tile(mask, (batchsize, 1, 1, 1))  # same masks for all images
    return xp.array(mask.reshape(batchsize, 1, imageHeight, imageWidth), dtype="float32") / 255.


def local_patch(x, bbox):
    """Crop local patch according to bbox.
    Args:
        x: input
        bbox: (top, left, height, width)
    """
    return x[:, :, bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]]


def resize_mask_like(mask, x):
    """Resize mask like shape of x.
    Args:
        mask: Original mask.
        x: To shape of x.
    """
    if mask.shape[2] < x.shape[2]:
        mask_resize = F.unpooling_2d(
            mask, ksize=4, outsize=x.shape[2:])
    else:
        rate = mask.shape[2] // x.shape[2]
        mask_resize = mask[:, :, ::rate, ::rate]
    return mask_resize


def spatial_discounting_mask(config, xp):
    """Generate spatial discounting mask constant.
    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.
    """
    gamma = config.SPATIAL_DISCOUNTING_GAMMA
    shape = [1, 1, config.HEIGHT, config.WIDTH]
    if config.DISCOUNTED_MASK:
        mask_values = np.ones((config.HEIGHT, config.WIDTH))
        for i in range(config.HEIGHT):
            for j in range(config.WIDTH):
                mask_values[i, j] = max(
                    gamma ** min(i, config.HEIGHT - i),
                    gamma ** min(j, config.WIDTH - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 1)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape)
    return xp.array(mask_values.astype("float32"))


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True, return_flow=False):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    """
    xp = cuda.get_array_module(f.data)
    # get shapes
    raw_fs = f.shape
    raw_int_fs = f.shape
    raw_int_bs = b.shape
    # extract patches from background with stride and rate
    kernel = 2 * rate
    pad = (kernel - rate * stride + 1) // 2
    raw_w = F.im2col(b, kernel, rate * stride, pad=pad).transpose(0, 2, 3, 1)
    raw_w = raw_w.reshape(raw_int_bs[0], -1, raw_int_bs[1], kernel, kernel)
    # raw_w = raw_w.transpose(0, 1, 4, 2, 3)  # transpose to b*hw*c*k*k
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = f[:, :, ::rate, ::rate]
    b = b[:, :, ::rate, ::rate]
    if mask is not None:
        mask = mask[:, :, ::rate, ::rate]
    fs = f.shape
    int_fs = f.shape
    f_groups = F.split_axis(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = b.shape
    int_bs = b.shape
    pad = (ksize - stride + 1) // 2
    w = F.im2col(b, ksize, stride, pad=pad).transpose(0, 2, 3, 1)
    w = w.reshape(int_fs[0], -1, int_fs[1], ksize, ksize)
    # w = w.transpose(0, 1, 4, 2, 3)  # transpose to b*hw*c*k*k
    # process mask
    if mask is None:
        mask = xp.zeros([1, 1, bs[2], bs[3]])
    m = F.im2col(mask, ksize, stride, pad=pad).transpose(0, 2, 3, 1).data
    m = m.reshape(1, -1, 1, ksize, ksize, )
    # m = m.transpose(0, 1, 4, 2, 3)  # transpose to b*hw*c*k*k
    # m = m[0]
    m = (m.mean(axis=(2, 3, 4)) == 0.).astype("float32").reshape(bs[0], 1, -1, 1, 1)
    w_groups = F.split_axis(w, int_bs[0], axis=0)
    raw_w_groups = F.split_axis(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = xp.eye(k).reshape(1, 1, k, k)
    for i, (xi, wi, raw_wi) in enumerate(zip(f_groups, w_groups, raw_w_groups)):
        # conv for compare
        wi = wi[0]
        mm = m[i]
        norm = F.sqrt(F.sum(F.square(wi), axis=(1, 2, 3), keepdims=True)) + 1e-4
        wi_normed = wi / (F.tile(norm, (1, *wi.shape[1:])))
        pad = (ksize) // 2
        yi = F.convolution_2d(xi, wi_normed, pad=pad)

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = yi.reshape(1, 1, fs[2] * fs[3], bs[2] * bs[3])
            pad = (fuse_k) // 2
            yi = F.convolution_2d(yi, fuse_weight, pad=pad)
            yi = yi.reshape(1, fs[2], fs[3], bs[2], bs[3])
            yi = yi.transpose(0, 2, 1, 4, 3)
            yi = yi.reshape(1, 1, fs[2] * fs[3], bs[2] * bs[3])
            yi = F.convolution_2d(yi, fuse_weight, pad=pad)
            yi = yi.reshape(1, fs[3], fs[2], bs[3], bs[2])
            yi = yi.transpose(0, 4, 3, 2, 1)
        yi = yi.reshape(1, bs[2] * bs[3], fs[2], fs[3])

        # softmax to match
        yi *= mm  # mask
        yi = F.softmax(yi * scale, 1)
        yi *= mm  # mask
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        pad = (kernel + rate * (yi.shape[2] - 1) - raw_fs[2]) // 2
        yi = F.deconvolution_2d(yi, wi_center, outsize=raw_fs[2:], stride=rate, pad=pad) / 4.
        y.append(yi)
        if return_flow:
            offset = xp.argmax(yi.data, axis=1)
            offset = xp.concatenate([offset // fs[2], offset % fs[2]], axis=0)
            offsets.append(offset)
    y = F.concat(y, axis=0).reshape(*raw_int_fs)
    if return_flow:
        offsets = xp.concatenate(offsets, axis=0).reshape(int_bs[0], 2, int_bs[2], int_bs[3])
        # case1: visualize optical flow: minus current position
        h_add = xp.tile(xp.reshape(xp.arange(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
        w_add = xp.tile(xp.reshape(xp.arange(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
        offsets = offsets - xp.concatenate([h_add, w_add], axis=3)
        # to flow image
        flow = flow_to_image_chainer(offsets)
        # # case2: visualize which pixels are attended
        if rate != 1:
            flow = F.unpooling_2d(flow, rate)
        return y, flow
    return y, None


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def flow_to_image(flow):
    """Transfer flow map to image.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_chainer(flow):
    """Chainer ops for computing flow to image.
    """
    img = flow_to_image(cuda.to_cpu(flow))
    img = img / 127.5 - 1.
    return img
