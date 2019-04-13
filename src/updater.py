import chainer


class GCUpdater(chainer.training.StandardUpdater):
    # updater for gated convolution inpainting
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model")
        self.config = kwargs.pop("config")
        super(GCUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        for i in range(self.config.DIS_UPDATE):
            batch_and_mask = self.get_iterator('main').next()
            batch_data, mask_data = zip(*batch_and_mask)
            batch_data = self.model.xp.array(list(batch_data))
            mask_data = self.model.xp.array(list(mask_data))
            losses = self.model.get_loss(batch_data, mask_data, calc_g_loss=(i == (self.config.DIS_UPDATE - 1)))
            self.model.discriminator.cleargrads()
            losses["d_loss"].backward()
            self.get_optimizer("d_opt").update()
        self.model.cleargrads()
        losses["g_loss"].backward()
        self.get_optimizer("g_opt").update()
        chainer.report({'ae_loss': losses["ae_loss"].data})
        chainer.report({'g_loss': losses["g_loss"].data})
        chainer.report({'d_loss': losses["d_loss"].data})


class CAUpdater(chainer.training.StandardUpdater):
    # updater for contextual attention inpainting
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model")
        self.config = kwargs.pop("config")
        super(CAUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        batch_data = self.get_iterator('main').next()
        batch_data = self.model.xp.array(batch_data)
        losses = self.model.get_loss(batch_data)
        self.model.cleargrads()
        losses["g_loss"].backward()
        self.get_optimizer("g_opt").update()

        self.model.cleargrads()
        losses["d_loss"].backward()
        self.get_optimizer("d_opt").update()

        chainer.report({'l1_loss': losses["l1_loss"]})
        chainer.report({'ae_loss': losses["ae_loss"]})
        chainer.report({'g_loss': losses["g_loss"]})
        chainer.report({'d_loss': losses["d_loss"]})
