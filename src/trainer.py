import os
import math
from decimal import Decimal

import torchvision.utils
from PIL import Image

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        epoch = self.optimizer.get_last_epoch() + 1
        if epoch <= self.args.first_stage_epoch:
            stage = 0
        elif epoch <= self.args.second_stage_epoch:
            stage = 1
        else:
            stage = 2
        self.loss[stage].step()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Stage {}][Epoch {}]\tLearning rate: {:.2e}'.format(stage+1, epoch, Decimal(lr))
        )
        self.loss[stage].start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, index,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            [lr_recons, hr_recons, lr_z, hr_z, sr] = self.model(lr, hr)
            loss = self.loss[stage](lr, hr, lr_recons, hr_recons, lr_z, hr_z, sr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()  # model parameters update

            timer_model.hold()

            # loss print
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss[stage].display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            # save results
            if self.args.save_train_results:
                if epoch % self.args.save_train_results_every == 0:
                    img_list = [lr, hr, lr_recons, hr_recons, lr_z, hr_z, sr]
                    # img_list = [lr, hr, lr_recons, hr_recons, sr]
                    self.ckp.save_train_results(stage, self.args.data_train, index, img_list, self.args.scale)

        self.loss[stage].end_log(len(self.loader_train))
        self.error_last = self.loss[stage].log[-1, -1]
        self.optimizer.schedule()  # learning rate update

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale)),
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_test_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    [lr_recons, hr_recons, lr_z, hr_z, sr] = self.model(lr, hr)
                    if self.args.normalized:
                        lr = utility.unNormalize(lr)
                        hr = utility.unNormalize(hr)
                        sr = utility.unNormalize(sr)
                    save_list = [sr]
                    self.ckp.psnr_log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.ssim_log[-1, idx_data, idx_scale] += utility.calc_ssim(
                        sr, hr, scale, self.args.rgb_range
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_test_results:
                        if epoch % self.args.save_test_results_every == 0:
                            self.ckp.save_test_results(d, filename[0], save_list, scale)

                self.ckp.psnr_log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.psnr_log.max(0)
                best_ssim = self.ckp.ssim_log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.psnr_log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        self.ckp.ssim_log[-1, idx_data, idx_scale],
                        best_ssim[0][idx_data, idx_scale],
                        best_ssim[1][idx_data, idx_scale] + 1,
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))

        if self.args.save_test_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
