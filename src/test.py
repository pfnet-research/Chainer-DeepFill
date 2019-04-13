import argparse, os
import chainer
from chainer import cuda, serializers
from PIL import Image

from inpaint_model import InpaintGCModel, InpaintCAModel
from config import Config
from dataset import Dataset

from utils import batch_postprocess_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='gated_convolution.yml',
                        help='path to config file')
    parser.add_argument('--model', type=str, default='v1',
                        help='model type: v1 or v2')
    parser.add_argument('--snapshot', type=str, default='model.npz',
                        help='path to snapshot')
    parser.add_argument('--name', type=str, default='result.png',
                        help='file name to save results')

    args = parser.parse_args()

    config = Config(args.config_path)

    if args.model == "v1":
        config.FREE_FORM = False
        inpaint_model = InpaintCAModel(config)
    elif args.model == "v2":
        inpaint_model = InpaintGCModel(config)
    else:
        assert False, "Model name '{args.model}' is invalid."

    if config.GPU_ID >= 0:
        chainer.cuda.get_device(config.GPU_ID).use()
        inpaint_model.to_gpu()

    if os.path.exists(args.snapshot):
        serializers.load_npz(args.snapshot, inpaint_model)
    else:
        assert False, "Flie '{args.snapshot}' does not exist."

    xp = inpaint_model.xp

    # training data
    test_dataset = Dataset(config, test=True, return_mask=True)
    test_iter = chainer.iterators.SerialIterator(test_dataset, 8)

    batch_and_mask = test_iter.next()
    batch_data, mask_data = zip(*batch_and_mask)
    batch_data = xp.array(batch_data)
    mask = xp.array(mask_data)

    batch_pos = batch_data / 127.5 - 1.
    # edges = None
    batch_incomplete = batch_pos * (1. - mask[:, :1])
    # inpaint
    x1, x2, offset_flow = inpaint_model.inpaintnet(
        batch_incomplete, mask, config)
    batch_complete = x2 * mask[:, :1] + batch_incomplete * (1. - mask[:, :1])
    # visualization
    viz_img = [batch_pos, batch_incomplete - mask[:, 1:] + mask[:, :1], batch_complete.data]

    batch_w = len(viz_img)
    batch_h = viz_img[0].shape[0]
    viz_img = xp.concatenate(viz_img, axis=0)
    viz_img = batch_postprocess_images(viz_img, batch_w, batch_h)
    viz_img = cuda.to_cpu(viz_img)
    Image.fromarray(viz_img).save(args.name)


if __name__ == '__main__':
    main()
