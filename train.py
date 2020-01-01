from model import *
from dataset import *

import itertools

from statistics import mean

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_cls = args.wgt_cls
        self.wgt_rec = args.wgt_rec
        self.wgt_gp = args.wgt_gp

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        self.nblk = args.nblk
        self.attrs = args.attrs
        self.ncls = len(self.attrs)

        self.ncritic = args.ncritic

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, netD, optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_cls = self.wgt_cls
        wgt_rec = self.wgt_rec
        wgt_gp = self.wgt_gp

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        attrs = self.attrs
        ncls = self.ncls

        ncritic = self.ncritic

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data)

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data)

        transform_train = transforms.Compose([CenterCrop((self.ny_load, self.nx_load)), Normalize(), RandomFlip(), Rescale((self.ny_in, self.nx_in)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(dir_data_train, data_type=self.data_type, transform=transform_train, attrs=attrs)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        # netG = UNet(nch_in + ncls, nch_out, nch_ker, norm)
        netG = ResNet(nch_in + ncls, nch_out, nch_ker, norm, nblk=self.nblk)
        netD = Discriminator(nch_out, nch_ker, norm, ncls=ncls, ny_in=self.ny_out, nx_in=self.nx_out)
        
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_REC = nn.L1Loss().to(device)   # L1
        fn_SRC = nn.BCEWithLogitsLoss().to(device)
        fn_GP = GradientPaneltyLoss().to(device)

        if self.name_data == 'celeba':
            fn_CLS = nn.BCEWithLogitsLoss().to(device)   # L1
        else:
            fn_CLS = nn.CrossEntropyLoss().to(device)  # L1

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))

        # optimG = torch.optim.Adam(itertools.chain(paramsG_a2b, paramsG_b2a), lr=lr_G, betas=(self.beta1, 0.999))
        # optimD = torch.optim.Adam(itertools.chain(paramsD_a, paramsD_b), lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, netD, optimG, optimD, st_epoch = self.load(dir_chck, netG, netD, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            loss_G_src_train = []
            loss_G_cls_train = []
            loss_G_rec_train = []

            loss_D_src_train = []
            loss_D_cls_train = []
            loss_D_gp_train = []

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = data[0]

                label_in = data[1]
                label_out = label_in[torch.randperm(label_in.size(0))]

                domain_in = get_domain(input, label_in)
                domain_out = get_domain(input, label_out)

                # Copy to GPU
                input = input.to(device)
                domain_in = domain_in.to(device)
                domain_out = domain_out.to(device)
                label_in = label_in.to(device)
                label_out = label_out.to(device)

                # forward netG
                output = netG(torch.cat([input, domain_out], dim=1))
                recon = netG(torch.cat([output, domain_in], dim=1))

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                src_in, cls_in = netD(input)
                src_out, cls_out = netD(output.detach())

                alpha = torch.rand(input.size(0), 1, 1, 1).to(self.device)
                output_ = (alpha * input + (1 - alpha) * output.detach()).requires_grad_(True)
                src_out_, _ = netD(output_)

                loss_D_src_in = fn_SRC(src_in, torch.ones_like(src_in))
                loss_D_src_out = fn_SRC(src_out, torch.zeros_like(src_out))
                loss_D_src = 0.5 * (loss_D_src_in + loss_D_src_out)

                loss_D_cls_in = fn_CLS(cls_in, label_in.view(label_in.size(0), label_in.size(1), 1, 1))
                loss_D_cls_out = fn_CLS(cls_out, label_out.view(label_out.size(0), label_out.size(1), 1, 1))
                loss_D_cls = 0.5 * (loss_D_cls_in + loss_D_cls_out)

                loss_D_gp = fn_GP(src_out_, output_)

                loss_D = loss_D_src + wgt_cls * loss_D_cls + wgt_gp * loss_D_gp
                loss_D.backward()
                optimD.step()

                # get losses
                loss_D_src_train += [loss_D_src.item()]
                loss_D_cls_train += [loss_D_cls.item()]
                loss_D_gp_train += [loss_D_gp.item()]

                if (i - 1) % ncritic == 0:
                    # backward netG
                    set_requires_grad(netD, False)
                    optimG.zero_grad()

                    src_out, cls_out = netD(output)

                    loss_G_src = fn_SRC(src_out, torch.ones_like(src_out))
                    loss_G_cls = fn_CLS(cls_out, label_out.view(label_out.size(0), label_out.size(1), 1, 1))
                    loss_G_rec = fn_REC(input, recon)

                    loss_G = loss_G_src + wgt_cls * loss_G_cls + wgt_rec * loss_G_rec

                    loss_G.backward()
                    optimG.step()

                    # get losses
                    loss_G_src_train += [loss_G_src.item()]
                    loss_G_cls_train += [loss_G_cls.item()]
                    loss_G_rec_train += [loss_G_rec.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'G_src: %.4f G_cls: %.4f G_rec: %.4f D_src: %.4f D_cls: %.4f D_gp: %.4f'
                      % (epoch, i, num_batch_train,
                         mean(loss_G_src_train), mean(loss_G_cls_train), mean(loss_G_rec_train),
                         mean(loss_D_src_train), mean(loss_D_cls_train), mean(loss_D_gp_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    output = transform_inv(output)
                    recon = transform_inv(recon)

                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output', output, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('recon', recon, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_G_src', mean(loss_G_src_train), epoch)
            writer_train.add_scalar('loss_G_cls', mean(loss_G_cls_train), epoch)
            writer_train.add_scalar('loss_G_rec', mean(loss_G_rec_train), epoch)
            writer_train.add_scalar('loss_D_src', mean(loss_D_src_train), epoch)
            writer_train.add_scalar('loss_D_cls', mean(loss_D_cls_train), epoch)
            writer_train.add_scalar('loss_D_gp', mean(loss_D_gp_train), epoch)

            # # update schduler
            # # schedG.step()
            # # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG, netD, optimG, optimD, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        dir_data_test = os.path.join(self.dir_data, self.name_data, 'test')

        transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        # netG_a2b = UNet(nch_in, nch_out, nch_ker, norm)
        # netG_b2a = UNet(nch_in, nch_out, nch_ker, norm)
        netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm, nblk=self.nblk)
        netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm, nblk=self.nblk)

        init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        netG_a2b, netG_b2a, st_epoch = self.load(dir_chck, netG_a2b, netG_b2a, mode=mode)

        ## test phase
        with torch.no_grad():
            netG_a2b.eval()
            netG_b2a.eval()
            # netG_a2b.train()
            # netG_b2a.train()

            gen_loss_l1_test = 0
            for i, data in enumerate(loader_test, 1):
                input_a = data['dataA'].to(device)
                input_b = data['dataB'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                input_a = transform_inv(input_a)
                input_b = transform_inv(input_b)
                output_a = transform_inv(output_a)
                output_b = transform_inv(output_b)
                recon_a = transform_inv(recon_a)
                recon_b = transform_inv(recon_b)

                for j in range(batch_size):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                               'input_a': "%04d-input_a.png" % name,
                               'input_b': "%04d-input_b.png" % name,
                               'output_a': "%04d-output_a.png" % name,
                               'output_b': "%04d-output_b.png" % name,
                               'recon_a': "%04d-recon_a.png" % name,
                               'recon_b': "%04d-recon_b.png" % name}

                    plt.imsave(os.path.join(dir_result_save, fileset['input_a']), input_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['input_b']), input_b[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output_a']), output_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output_b']), output_b[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['recon_a']), recon_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['recon_b']), recon_b[j, :, :, :].squeeze())

                    append_index(dir_result, fileset)

                    print("%d / %d" % (name + 1, num_test))


def get_domain(input, label):
    domain = label.clone()
    domain = domain.view((domain.size(0), domain.size(1), 1, 1))
    domain = domain.repeat(1, 1, input.size(2), input.size(3))

    return domain


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input_a</th><th>output_b</th><th>recon_a</th><th>input_b</th><th>output_a</th><th>recon_b</th></tr>")

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    for kind in ["input_a", "output_b", "recon_a", "input_b", "output_a", "recon_b"]:
        index.write("<td><img src='images/%s'></td>" % fileset[kind])

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
