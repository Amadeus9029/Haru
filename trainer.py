import os
import math
from decimal import Decimal
import time
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from torch.autograd import Variable
from data.helper import *
from tensorboard_logger import configure, log_value


class HaruTrainer():
    def __init__(self, data, loader, my_model, my_loss, ckp, args=None):
        # 肯定需要数据集，损失函数，模型,权重，lr的调整
        self.args = args
        self.scale = args.scale
        self.data = data
        self.ckp = ckp

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train_stage1(self):
        for epoch in range(1, self.epoch_limit + 1):
            print("Epoch: %d: " % epoch)
            self.epoch = epoch
            self.train_one_epoch_stage1()
            self.validate()
            self.predict_resize(repr(epoch))
            state = {'epoch': self.epoch, 'G': self.G.state_dict(),
                     'best_valid_acc': self.best_valid_acc, 'lr': self.LR}
            self.save_checkpoint(state, 'last', False)
            if epoch % 2 == 0:
                self.save_checkpoint(state, '-' + str(epoch), False)
            if epoch % 10 == 0:
                self.LR = self.LR / 2
                self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.LR)

    def train_stage2(self):
        print("Using GPU: #", self.gpuid)
        self.load_checkpoint(self.pretrained_weights, self.parallel, best=False, load_lr=False)
        self.G.cuda(self.gpuid)
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.LR)
        self.set_gradients(False)
        init_epoch = self.epoch
        self.predict_resize(repr(0))
        for epoch in range(init_epoch, self.epoch_limit + 1):
            print("Epoch: %d: " % epoch)
            # reset discriminator
            self.D.apply(self.init_weights)
            self.epoch = epoch
            self.train_one_epoch_stage2()
            self.validate()
            self.predict_resize(repr(epoch))
            state = {'epoch': self.epoch, 'G': self.G.state_dict(), 'D': self.D.state_dict(),
                     'best_valid_acc': self.best_valid_acc, 'lr': self.LR}
            self.save_checkpoint(state, 'last', False)
            if epoch % 2 == 0:
                self.save_checkpoint(state, '-' + str(epoch), False)
            if epoch % 10 == 0:
                self.set_gradients(True)
                self.trainable(self.D, True)
                self.LR = self.LR / 2
                self.D_optim = torch.optim.Adam(self.D.parameters(), lr=self.LR, betas=(0.5, 0.999))
                self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.LR * 0.1, betas=(0.5, 0.999))
                self.set_gradients(False)

    def train_one_epoch_stage1(self):
        epoch_loss = 0
        tic = time.time()
        # log losses
        losses = AverageMeter()
        trans_losses = AverageMeter()
        atm_losses = AverageMeter()
        streak_losses = AverageMeter()
        accs = AverageMeter()
        atmval = AverageMeter()
        gr_losses = AverageMeter()
        clean_losses = AverageMeter()

        # dataloader = self.load_data('train', aug=False)
        train_sample_len = len(self.data)
        with tqdm(total=len(self.data) * self.batch_size) as pbar:
            for i, self.input_list in enumerate(self.data):
                # input_list: rain, st_sp, st_md, st_ds, im_sp, im_md, im_ds, mask(3 channel)
                image_in_var = Variable(self.input_list[0]).cuda(self.gpuid)
                streak_gt_var = Variable(self.input_list[1]).cuda(self.gpuid)
                trans_gt_var = Variable(self.input_list[2]).cuda(self.gpuid)
                atm_gt_var = Variable(self.input_list[3]).cuda(self.gpuid)
                clean_gt_var = Variable(self.input_list[4]).cuda(self.gpuid)

                # forward
                # NOTE : self.st_out to be added
                self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(image_in_var)

                # compute loss
                loss_sp = self.criterionMSE(self.st_out, streak_gt_var)
                loss_tr = self.criterionMSE(self.trans_out, trans_gt_var)
                loss_atm = self.criterionMSE(self.atm_out, atm_gt_var)
                loss_clean = self.criterionMSE(self.clean_out, clean_gt_var)
                loss_pc = self.criterionMSE(self.vgg(self.clean_out, 3), self.vgg(clean_gt_var, 3))
                gradient_h_est, gradient_v_est = gradient(self.trans_out)
                gradient_h_gt, gradient_v_gt = gradient(trans_gt_var)
                loss_trans_gradient_h = self.criterionL1(gradient_h_est, gradient_h_gt)
                loss_trans_gradient_v = self.criterionL1(gradient_v_est, gradient_v_gt)
                loss_gradient = loss_trans_gradient_h + loss_trans_gradient_v

                self.total_loss = loss_sp + loss_tr + loss_atm + 0.5 * loss_gradient
                epoch_loss += self.total_loss.item()
                losses.update(self.total_loss.item(), self.batch_size)
                atmval.update(torch.mean(self.atm_out), self.batch_size)
                atm_losses.update(loss_atm.item(), self.batch_size)
                trans_losses.update(loss_tr.item(), self.batch_size)
                streak_losses.update(loss_sp.item(), self.batch_size)
                clean_losses.update(loss_clean.item(), self.batch_size)
                gr_losses.update(loss_pc.item(), self.batch_size)

                # backward
                self.G_optim.zero_grad()
                self.total_loss.backward()
                self.G_optim.step()

                # logging
                toc = time.time()
                pbar.set_description(
                    (
                        "{:.1f}s L:{:.4f} sp:{:.4f} tr:{:.4f} atm:{:.4f} im:{:.4f} gr:{:.4f} LR:{:.6f} acc:{:.2f} )".format(
                            (toc - tic),
                            self.total_loss.item(),
                            loss_sp.item(),
                            loss_tr.item(),
                            loss_atm.item(),
                            loss_clean.item(),
                            loss_gradient.item(),
                            self.LR, accs.avg)
                    )
                )
                pbar.update(self.batch_size)

                # == Evaluation Region == #
                recons = (image_in_var - (1 - self.trans_out) * self.atm_out) / (self.trans_out + 0.0001) - self.st_out
                mini_acc = compute_psnr(recons, clean_gt_var)
                accs.update(mini_acc, self.batch_size)

                # write output
                if i % 10 == 0:
                    self.write_image_stage1('./out.jpg')

                if self.use_tensorboard:
                    iteration = (self.epoch - 1) * train_sample_len + i
                    log_value('train_loss', losses.avg, iteration)
                    log_value('train_acc', accs.avg, iteration)
                    log_value('atm_loss', atm_losses.avg, iteration)
                    log_value('trans_loss', trans_losses.avg, iteration)
                    log_value('streak_loss', streak_losses.avg, iteration)
                    log_value('atm_value', atmval.avg, iteration)
                    log_value('clean_loss', clean_losses.avg, iteration)
                    log_value('gr_loss', gr_losses.avg, iteration)

            print("Total Loss: %f" % epoch_loss)

    def train_one_epoch_stage2(self):
        epoch_loss = 0
        tic = time.time()
        toc = tic
        imloss = AverageMeter()
        pcloss = AverageMeter()
        genloss = AverageMeter()
        disloss = AverageMeter()
        Dtrueloss = AverageMeter()
        Dfakeloss = AverageMeter()
        Gadvloss = AverageMeter()
        Gadvlossrain = AverageMeter()
        dataloader = self.load_data('train', aug=False)
        self.train_sample_len = len(dataloader)
        with tqdm(total=len(dataloader) * self.batch_size) as pbar:
            for i, self.input_list in enumerate(dataloader):
                if np.random.rand() <= 0.1:
                    self.real_synt_toggler = 1  # for real rain images
                else:
                    self.real_synt_toggler = 0  # for synthetic rain images

                # input_list: rain, st_sp, st_md, st_ds, im_sp, im_md, im_ds, mask(3 channel)
                self.image_in_var = Variable(self.input_list[0]).cuda(self.gpuid)
                self.streak_gt_var = Variable(self.input_list[1]).cuda(self.gpuid)
                self.trans_gt_var = Variable(self.input_list[2]).cuda(self.gpuid)
                self.atm_gt_var = Variable(self.input_list[3]).cuda(self.gpuid)
                self.clean_gt_var = Variable(self.input_list[4]).cuda(self.gpuid)
                self.realrain_gt_var = Variable(self.input_list[5]).cuda(self.gpuid)

                # DISCRIMINATOR
                self.trainable(self.D, True)
                self.D.zero_grad()
                self.train_dis()  # real error and fake error backward() together
                self.D_optim.step()
                self.trainable(self.D, False)

                # GENERATOR
                self.G.zero_grad()
                self.train_gen()
                self.G_optim.step()

                # write output
                if i % 10 == 0:
                    self.write_image_stage2('./out.jpg')

                # LOG LOSS
                pbar.set_description(
                    (
                        "{:.1f}s L:{:.4f} im:{:.4f} adv:{:.4f} gr:{:.4f} pc:{:4f} rain:{:4f} tl:{:4f} fl:{:4f} prb:{:.3f} LR:{:.6f} acc:{:.2f} )".format(
                            (toc - tic),
                            self.total_loss.item(),
                            self.loss_clean.item(),
                            self.loss_adv.item(),
                            self.loss_gradient.item(),
                            self.loss_pc.item(),
                            self.loss_adv_realrain.item(),
                            self.tl.item(), self.fl.item(),
                            self.probability.item(),
                            self.LR, self.accs.avg)
                    )
                )
                pbar.update(self.batch_size)
                epoch_loss += self.total_loss.item()
                imloss.update(self.loss_clean.item(), self.batch_size)
                pcloss.update(self.loss_pc.item(), self.batch_size)
                genloss.update(self.total_loss.item(), self.batch_size)
                disloss.update(self.probability.item(), self.batch_size)
                Dtrueloss.update(self.tl.item(), self.batch_size)
                Dfakeloss.update(self.fl.item(), self.batch_size)
                Gadvloss.update(self.loss_adv.item(), self.batch_size)
                Gadvlossrain.update(self.loss_adv_realrain.item(), self.batch_size)

                # logging
                toc = time.time()
                if self.use_tensorboard:
                    iteration = (self.epoch - 1) * self.train_sample_len / self.batch_size + i
                    log_value('dis_loss', disloss.avg, iteration)
                    log_value('gen_loss', genloss.avg, iteration)
                    log_value('acc', self.accs.avg, iteration)
                    log_value('D_true_loss', Dtrueloss.avg, iteration)
                    log_value('D_fake_loss', Dfakeloss.avg, iteration)
                    log_value('G_adv_loss', Gadvloss.avg, iteration)
                    log_value('G_adv_realrain_loss', Gadvlossrain.avg, iteration)

            print("Total Loss: %f" % epoch_loss)

    def train_dis(self):
        self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(self.image_in_var)
        # i. real data input:clean ground truth. ii. fake data input: output of generator  G(image_in_var)
        # 1. Train D on real data
        depth_gt_real = Variable(torch.zeros(self.batch_size, 1, self.image_size, self.image_size)).cuda(self.gpuid)
        d_realdata_input = torch.cat((self.image_in_var, self.clean_gt_var), dim=1)
        depth_real, d_realdata_output = self.D(d_realdata_input)  # result should be True (1)
        # depth_real = depth_real.repeat(1,3,1,1)
        d_realdata_error = self.criterionGAN(d_realdata_output, True).cuda(self.gpuid)
        d_realdepth_error = self.criterionMSE(depth_real, depth_gt_real)
        total_loss = d_realdata_error + d_realdepth_error

        # 2. Train D on fake data

        d_fakedata_input = torch.cat((self.image_in_var, self.clean_out), dim=1)
        depth_fake, d_fakedata_output = self.D(d_fakedata_input.detach())
        d_fakedata_error = self.criterionGAN(d_fakedata_output, False)
        depth_fake = depth_fake.repeat(1, 3, 1, 1)
        d_fakedepth_error = self.criterionMSE(depth_fake, 1 - self.trans_out.detach())
        total_loss += d_fakedata_error + d_fakedepth_error
        total_loss.backward()

        self.probability = (d_realdata_error + d_fakedata_error).mean()
        self.fl = d_fakedata_error
        self.tl = d_realdata_error

    def train_gen(self):
        # Feed Forward
        self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(self.image_in_var)
        D_input = torch.cat((self.image_in_var, self.clean_out), dim=1)
        depth_mask, self.dis_out = self.D(D_input.detach())

        # compute loss
        self.loss_clean = self.criterionMSE(self.clean_out, self.clean_gt_var)
        self.loss_adv = self.criterionGAN(self.dis_out, True)

        # get gradients
        gradient_h_est, gradient_v_est = gradient(self.clean_out)
        gradient_h_gt, gradient_v_gt = gradient(self.clean_gt_var)
        loss_trans_gradient_h = self.criterionL1(gradient_h_est, gradient_h_gt)
        loss_trans_gradient_v = self.criterionL1(gradient_v_est, gradient_v_gt)
        self.loss_gradient = loss_trans_gradient_h + loss_trans_gradient_v
        self.loss_pc = self.criterionMSE(self.vgg(self.clean_out, 8), self.vgg(self.clean_gt_var, 8))
        self.realrain_st, self.realrain_trans, self.realrain_atm, self.realrain_out = self.G(self.realrain_gt_var)

        # sum loss
        if self.real_synt_toggler == 1:
            # print("Real Rain!")
            realrain_D_input = torch.cat((self.realrain_gt_var, self.realrain_out), dim=1)
            depth_mask, self.dis_realrain_out = self.D(realrain_D_input.detach())
            self.loss_adv_realrain = self.criterionGAN(self.dis_realrain_out, True)
            self.total_loss = self.loss_clean + 0.01 * self.loss_adv + 2 * self.loss_pc + self.loss_gradient + 0.01 * self.loss_adv_realrain
        else:
            self.total_loss = self.loss_clean + 0.01 * self.loss_adv + 2 * self.loss_pc + self.loss_gradient

        # backward
        self.total_loss.backward()

        # # == Evaluation Region == #
        mini_acc = compute_psnr(self.clean_out, self.clean_gt_var)
        self.accs.update(mini_acc, self.batch_size)

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
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
