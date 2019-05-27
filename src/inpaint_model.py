import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
from PIL import Image

from inpaint_ops import GenConv, GenDeconv, DisConv
from inpaint_ops import random_bbox, bbox2mask, local_patch, free_form_mask
from inpaint_ops import spatial_discounting_mask
from inpaint_ops import resize_mask_like, contextual_attention

from utils import batch_postprocess_images


class InpaintNet(chainer.Chain):
    def __init__(self, gated=False):
        cnum = 32
        self.gated = gated
        super(InpaintNet, self).__init__()
        with self.init_scope():
            self.conv1 = GenConv(5, cnum, 5, 1, gated=gated)
            self.conv2_downsample = GenConv(cnum, 2 * cnum, 3, 2, gated=gated)
            self.conv3 = GenConv(2 * cnum, 2 * cnum, 3, 1, gated=gated)
            self.conv4_downsample = GenConv(2 * cnum, 4 * cnum, 3, 2, gated=gated)
            self.conv5 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.conv6 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.conv7_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=2, gated=gated)
            self.conv8_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=4, gated=gated)
            self.conv9_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=8, gated=gated)
            self.conv10_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=16, gated=gated)
            self.conv11 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.conv12 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.conv13_upsample = GenDeconv(4 * cnum, 2 * cnum, gated=gated)
            self.conv14 = GenConv(2 * cnum, 2 * cnum, 3, 1, gated=gated)
            self.conv15_upsample = GenDeconv(2 * cnum, cnum, gated=gated)
            self.conv16 = GenConv(cnum, cnum // 2, 3, 1, gated=gated)
            self.conv17 = GenConv(cnum // 2, 3, 3, 1, activation=None)

            # conv branch
            self.xconv1 = GenConv(5, cnum, 5, 1, gated=gated)
            self.xconv2_downsample = GenConv(cnum, cnum, 3, 2, gated=gated)
            self.xconv3 = GenConv(cnum, 2 * cnum, 3, 1, gated=gated)
            self.xconv4_downsample = GenConv(2 * cnum, 2 * cnum, 3, 2, gated=gated)
            self.xconv5 = GenConv(2 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.xconv6 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.xconv7_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=2, gated=gated)
            self.xconv8_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=4, gated=gated)
            self.xconv9_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=8, gated=gated)
            self.xconv10_atrous = GenConv(4 * cnum, 4 * cnum, 3, rate=16, gated=gated)

            # attention branch
            self.pmconv1 = GenConv(5, cnum, 5, 1, gated=gated)
            self.pmconv2_downsample = GenConv(cnum, cnum, 3, 2, gated=gated)
            self.pmconv3 = GenConv(cnum, 2 * cnum, 3, 1, gated=gated)
            self.pmconv4_downsample = GenConv(2 * cnum, 4 * cnum, 3, 2, gated=gated)
            self.pmconv5 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.pmconv6 = GenConv(4 * cnum, 4 * cnum, 3, 1, activation="relu", gated=gated)
            self.pmconv9 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.pmconv10 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)

            self.allconv11 = GenConv(8 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.allconv12 = GenConv(4 * cnum, 4 * cnum, 3, 1, gated=gated)
            self.allconv13_upsample = GenDeconv(4 * cnum, 2 * cnum, gated=gated)
            self.allconv14 = GenConv(2 * cnum, 2 * cnum, 3, 1, gated=gated)
            self.allconv15_upsample = GenDeconv(2 * cnum, cnum, gated=gated)
            self.allconv16 = GenConv(cnum, cnum // 2, 3, 1, gated=gated)
            self.allconv17 = GenConv(cnum // 2, 3, 3, 1, activation=None)

    def __call__(self, x, mask, return_offset=False):
        xin = x
        if mask.shape[1] == 1:  # no edge image
            mask = F.concat([mask, self.xp.zeros_like(x[:, :1])])

        x = F.concat([x, mask], axis=1)

        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask_s = resize_mask_like(mask[:, :1], x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = F.clip(x, -1., 1.)
        x_stage1 = x

        # stage2, paste result as input
        x = x * mask[:, :1] + xin * (1. - mask[:, :1])
        # conv branch
        xnow = F.concat([x, mask], axis=1)
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)

        x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2, return_flow=return_offset)
        x = self.pmconv9(x)
        x = self.pmconv10(x)

        # pm = x
        x = F.concat([x_hallu, x], axis=1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)
        x = self.allconv17(x)

        x_stage2 = F.clip(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow


class WganLocalDiscriminator(chainer.Chain):
    def __init__(self):
        cnum = 64
        super(WganLocalDiscriminator, self).__init__()
        with self.init_scope():
            self.conv1 = DisConv(3, cnum)
            self.conv2 = DisConv(cnum, 2 * cnum)
            self.conv3 = DisConv(2 * cnum, 4 * cnum)
            self.conv4 = DisConv(4 * cnum, 8 * cnum)
            self.linear = L.Linear(None, 1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear(x)
        return x


class WganGlobalDiscriminator(chainer.Chain):
    def __init__(self):
        cnum = 64
        super(WganGlobalDiscriminator, self).__init__()
        with self.init_scope():
            self.conv1 = DisConv(3, cnum)
            self.conv2 = DisConv(cnum, 2 * cnum)
            self.conv3 = DisConv(2 * cnum, 4 * cnum)
            self.conv4 = DisConv(4 * cnum, 4 * cnum)
            self.linear = L.Linear(None, 1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear(x)
        return x


class WganDisciriminatoir(chainer.Chain):
    def __init__(self):
        super(WganDisciriminatoir, self).__init__()
        with self.init_scope():
            self.wgan_local_discriminator = WganLocalDiscriminator()
            self.wgan_global_discriminator = WganGlobalDiscriminator()

    def __call__(self, batch_local, batch_global):
        dout_local = self.wgan_local_discriminator(batch_local)
        dout_global = self.wgan_global_discriminator(batch_global)
        return dout_local, dout_global


class SnPatchGanDiscriminator(chainer.Chain):
    # size of receptive field is 253x253
    def __init__(self):
        cnum = 64
        super(SnPatchGanDiscriminator, self).__init__()
        with self.init_scope():
            self.conv1 = DisConv(None, cnum, sn=True)
            self.conv2 = DisConv(cnum, 2 * cnum, sn=True)
            self.conv3 = DisConv(2 * cnum, 4 * cnum, sn=True)
            self.conv4 = DisConv(4 * cnum, 4 * cnum, sn=True)
            self.conv5 = DisConv(4 * cnum, 4 * cnum, sn=True)
            self.conv6 = DisConv(4 * cnum, 4 * cnum, sn=True)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class InpaintCAModel(chainer.Chain):
    def __init__(self, config):
        super(InpaintCAModel, self).__init__()
        with self.init_scope():
            self.inpaintnet = InpaintNet()
            self.discriminator = WganDisciriminatoir()
        self.config = config

    def get_loss(self, batch_data):
        config = self.config
        batch_pos = batch_data / 127.5 - 1
        bbox = random_bbox(config)
        mask = bbox2mask(bbox, batch_data.shape[0], config, self.xp)
        batch_incomplete = batch_pos * (1 - mask)
        x1, x2, offset_flow = self.inpaintnet(batch_incomplete, mask, config)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
        else:
            batch_predicted = x2
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted * mask + batch_incomplete * (1 - mask)
        # local patches
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        local_patch_x1 = local_patch(x1, bbox)
        local_patch_x2 = local_patch(x2, bbox)
        local_patch_batch_complete = local_patch(batch_complete, bbox)
        local_patch_mask = local_patch(mask, bbox)
        l1_alpha = config.COARSE_L1_ALPHA
        losses["l1_loss"] = l1_alpha * F.mean(F.absolute(local_patch_batch_pos - local_patch_x1) *
                                              spatial_discounting_mask(config, self.xp))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['l1_loss'] += F.mean(F.absolute(local_patch_batch_pos - local_patch_x2) *
                                        spatial_discounting_mask(config, self.xp))
        losses['ae_loss'] = l1_alpha * F.mean(F.absolute(batch_pos - x1) * (1. - mask))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['ae_loss'] += F.mean(F.absolute(batch_pos - x2) * (1. - mask))
        losses['ae_loss'] /= F.mean(1. - mask)

        # gan
        batch_pos_neg = F.concat([batch_pos, batch_complete], axis=0)
        # local deterministic patch
        local_patch_batch_pos_neg = F.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = F.concat([batch_pos_neg, mask], axis=1)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.discriminator(local_patch_batch_pos_neg, batch_pos_neg)
            pos_local, neg_local = F.split_axis(pos_neg_local, 2, axis=0)
            pos_global, neg_global = F.split_axis(pos_neg_global, 2, axis=0)
            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local)
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global)
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.discriminator(interpolates_local, interpolates_global)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']

        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
        return losses

    def evaluation(self, test_image_folder):
        config = self.config

        @chainer.training.make_extension()
        def evaluation(trainer):
            it = trainer.updater.get_iterator('test')
            batch_data = it.next()
            batch_data = self.xp.array(batch_data)

            # generate mask, 1 represents masked point
            bbox = (config.HEIGHT // 2, config.WIDTH // 2,
                    config.HEIGHT, config.WIDTH)
            config.MAX_DELTA_HEIGHT = 0
            config.MAX_DELTA_WIDTH = 0
            if bbox is None:
                bbox = random_bbox(config)
            mask = bbox2mask(bbox, batch_data.shape[0], config, self.xp)
            batch_pos = batch_data / 127.5 - 1.
            batch_incomplete = batch_pos * (1. - mask)
            # inpaint
            x1, x2, offset_flow = self.inpaintnet(
                batch_incomplete, mask, config)
            if config.PRETRAIN_COARSE_NETWORK:
                batch_predicted = x1
            else:
                batch_predicted = x2
            # apply mask and reconstruct
            batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
            # visualization
            viz_img = [batch_pos, batch_incomplete + mask, batch_complete.data]
            if offset_flow is not None:
                viz_img.append(
                    F.unpooling_2d(offset_flow, 4).data)
            batch_w = len(viz_img)
            batch_h = viz_img[0].shape[0]
            viz_img = self.xp.concatenate(viz_img, axis=0)
            viz_img = batch_postprocess_images(viz_img, batch_w, batch_h)
            viz_img = cuda.to_cpu(viz_img)
            Image.fromarray(viz_img).save(test_image_folder + "/iter_" + str(trainer.updater.iteration) + ".jpg")

        return evaluation


class InpaintGCModel(chainer.Chain):
    def __init__(self, config):
        super(InpaintGCModel, self).__init__()
        with self.init_scope():
            self.inpaintnet = InpaintNet(gated=True)
            self.discriminator = SnPatchGanDiscriminator()
        self.config = config

    def get_loss(self, batch_data, mask, calc_g_loss=True):
        config = self.config
        batch_pos = batch_data / 127.5 - 1

        losses = {}
        batch_incomplete = batch_pos * (1 - mask[:, :1])

        chainer.config.enable_backprop = calc_g_loss  # prevent to calc grad of generator
        x1, x2, offset_flow = self.inpaintnet(batch_incomplete, mask, config)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            # print("set batch_predicted to x1")
        else:
            batch_predicted = x2
            # print("set batch_predicted to x2")
        # apply mask and complete image
        batch_complete = batch_predicted * mask[:, :1] + batch_incomplete * (1 - mask[:, :1])

        if calc_g_loss:
            l1_alpha = config.COARSE_L1_ALPHA
            losses['ae_loss'] = l1_alpha * F.mean(F.absolute(batch_pos - x1))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['ae_loss'] += F.mean(F.absolute(batch_pos - x2))

        # gan
        batch_pos = self.xp.concatenate([batch_pos, mask], axis=1)
        batch_complete = F.concat([batch_complete, mask], axis=1)

        batch_pos_neg = F.concat([batch_pos, batch_complete], axis=0)

        chainer.config.enable_backprop = True
        # sn-patch gan
        pos_neg_global = self.discriminator(batch_pos_neg)
        pos_global, neg_global = F.split_axis(pos_neg_global, 2, axis=0)
        # sngan loss
        g_loss_global, d_loss_global = gan_sngan_loss(pos_global, neg_global, d_loss_only=not calc_g_loss)
        losses['g_loss'] = g_loss_global + losses['ae_loss'] if calc_g_loss else None
        losses['d_loss'] = d_loss_global

        return losses

    def evaluation(self, test_image_folder):
        config = self.config

        @chainer.training.make_extension()
        def evaluation(trainer):
            it = trainer.updater.get_iterator('test')
            batch_and_mask = it.next()
            batch_data, mask_data = zip(*batch_and_mask)
            batch_data = self.xp.array(batch_data)
            mask = self.xp.array(mask_data)

            batch_pos = batch_data / 127.5 - 1.
            # edges = None
            batch_incomplete = batch_pos * (1. - mask[:, :1])
            # inpaint
            x1, x2, offset_flow = self.inpaintnet(
                batch_incomplete, mask, config)
            if config.PRETRAIN_COARSE_NETWORK:
                batch_predicted = x1
                # logger.info('Set batch_predicted to x1.')
            else:
                batch_predicted = x2
                # logger.info('Set batch_predicted to x2.')
            # apply mask and reconstruct
            batch_complete = batch_predicted * mask[:, :1] + batch_incomplete * (1. - mask[:, :1])
            # visualization
            viz_img = [batch_pos, batch_incomplete - mask[:, 1:] + mask[:, :1], batch_complete.data]
            if offset_flow is not None:
                viz_img.append(
                    F.unpooling_2d(offset_flow, 4).data)
            batch_w = len(viz_img)
            batch_h = viz_img[0].shape[0]
            viz_img = self.xp.concatenate(viz_img, axis=0)
            viz_img = batch_postprocess_images(viz_img, batch_w, batch_h)
            viz_img = cuda.to_cpu(viz_img)
            Image.fromarray(viz_img).save(test_image_folder + "/iter_" + str(trainer.updater.iteration) + ".jpg")

        return evaluation


def gan_wgan_loss(pos, neg):
    # """
    # wgan loss function for GANs.
    # - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    # """
    d_loss = F.mean(neg - pos)
    g_loss = -F.mean(neg)
    return g_loss, d_loss


def gan_sngan_loss(pos, neg, d_loss_only=False):
    # SN-PatchGAN loss with hinge loss
    d_loss = F.mean(F.relu(1 - pos) + F.relu(1 + neg))
    g_loss = None if d_loss_only else -F.mean(neg)
    return g_loss, d_loss


def gradients_penalty(x, y, mask=None, norm=1.):
    # """Improved Training of Wasserstein GANs
    # - https://arxiv.org/abs/1704.00028
    # """

    gradients = chainer.grad([y], [x], enable_double_backprop=True)[0]
    if mask is None:
        xp = cuda.get_array_module(x)
        mask = xp.ones_like(x)
    slopes = F.sqrt(F.sum(gradients ** 2 * mask, axis=(1, 2, 3)))
    return F.mean(F.square(slopes - norm))


def random_interpolates(x, y, alpha=None):
    # """
    # x: first dimension as batch_size
    # y: first dimension as batch_size
    # alpha: [BATCH_SIZE, 1]
    # """
    shape = x.shape
    if alpha is None:
        xp = cuda.get_array_module(x)
        alpha = xp.random.uniform(size=(shape[0], 1, 1, 1))
    interpolates = x + (y - x) * alpha
    return F.reshape(interpolates, shape)
