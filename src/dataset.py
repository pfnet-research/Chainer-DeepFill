import chainer
import numpy as np
import random
from inpaint_ops import bbox2mask, free_form_mask, random_bbox


def _create_mask(config, edge=None):
    # num_class = 1  # only for person.
    size = config.IMG_SHAPES[:2]
    if config.FREE_FORM:
        mask = free_form_mask(np, 1, size)
    else:
        bbox = random_bbox(config)
        mask = bbox2mask(bbox, 1, config, np)
    if edge is not None:
        edge = mask * (edge[None, :1] / 255.)
        mask = np.concatenate([mask, edge], axis=1)

    return mask[0]


def _postprocess_image(image, edge, config):
    # random crop
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    height, width = config.IMG_SHAPES[:2]
    h, w = image.shape[1:3]
    offset = (random.randint(0, h - height), random.randint(0, w - width))
    image = image[:, offset[0]:offset[0] + height, offset[1]:offset[1] + width]
    if edge is not None:
        edge = edge[:, offset[0]:offset[0] + height, offset[1]:offset[1] + width]
        return image, edge
    return image, None


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, config, return_mask=False, test=False):
        """
        dataset for image inpainting
        :param config:
        :param test: True if this dataset used for test
        """
        dataset_id = int(test)
        self.image_dataset = chainer.datasets.ImageDataset(config.IMAGE_FLIST[dataset_id])
        if config.EDGE_FLIST:
            self.edge_dataset = chainer.datasets.ImageDataset(config.EDGE_FLIST[dataset_id])
        self.config = config
        self.return_mask = return_mask

    def __len__(self):
        return len(self.image_dataset)

    def get_example(self, i):
        image = self.image_dataset[i]
        edge = self.edge_dataset[i] if self.return_mask and self.config.EDGE_FLIST else None
        image, edge = _postprocess_image(image, edge, self.config)
        if self.return_mask:
            if edge is None:
                edge = np.zeros_like(image[:1])
            return image, _create_mask(self.config, edge)
        return image
