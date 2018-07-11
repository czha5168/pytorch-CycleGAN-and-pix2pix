import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixBraTS17MultipleOutputsModel(BaseModel):
    def name(self):
        return 'Pix2PixBraTS17MultipleOutputsModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256_multiple_outputs')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        #self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN', 'G_L1', 'D_to1', 'D_to2', 'D_to3']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_from', 'fake_to1', 'real_to1', 'fake_to2', 'real_to2', 'fake_to3', 'real_to3']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D_to1', 'D_to2', 'D_to3']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.n_outbranches)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_to1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_to2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_to3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_to1_pool = ImagePool(opt.pool_size)
            self.fake_to2_pool = ImagePool(opt.pool_size)
            self.fake_to3_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_to1.parameters(),
                                                                self.netD_to2.parameters(),
                                                                self.netD_to3.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        from_modality = self.opt.which_direction
        if from_modality == 'FROM_T1':
            self.real_from = input['T1'].to(self.device)
            self.real_to1 = input['T2'].to(self.device)
            self.real_to2 = input['T1CE'].to(self.device)
            self.real_to3 = input['FLAIR'].to(self.device)
        elif from_modality == 'FROM_T2':
            self.real_from = input['T2'].to(self.device)
            self.real_to1 = input['T1'].to(self.device)
            self.real_to2 = input['T1CE'].to(self.device)
            self.real_to3 = input['FLAIR'].to(self.device)
        elif from_modality == 'FROM_T1CE':
            self.real_from = input['T1CE'].to(self.device)
            self.real_to1 = input['T1'].to(self.device)
            self.real_to2 = input['T2'].to(self.device)
            self.real_to3 = input['FLAIR'].to(self.device)
        elif from_modality == 'FROM_FLAIR':
            self.real_from = input['FLAIR'].to(self.device)
            self.real_to1 = input['T1'].to(self.device)
            self.real_to2 = input['T2'].to(self.device)
            self.real_to3 = input['T1CE'].to(self.device)
        else:
            print('wrong setting for "which_direction"')
            exit(1)
        self.image_paths = input['all_in_one_path']

    def forward(self):
        self.fake_to1, self.fake_to2, self.fake_to3 = self.netG(self.real_from)
    def backward_D_basic(self, netD, fake_from_to, real_from_to):
        # real
        pred_real = netD(real_from_to)
        loss_D_real = self.criterionGAN(pred_real, True)
        # fake
        pred_fake = netD(fake_from_to.detach())    # Detach - important
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # combine
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    def backward_D_to1(self):
        fake_from_to1 = self.fake_to1_pool.query(torch.cat((self.real_from, self.fake_to1), 1))
        real_from_to1 = torch.cat((self.real_from, self.real_to1), 1)
        self.loss_D_to1 = self.backward_D_basic(self.netD_to1, fake_from_to1, real_from_to1)
    def backward_D_to2(self):
        fake_from_to2 = self.fake_to2_pool.query(torch.cat((self.real_from, self.fake_to2), 1))
        real_from_to2 = torch.cat((self.real_from, self.real_to2), 1)
        self.loss_D_to2 = self.backward_D_basic(self.netD_to2, fake_from_to2, real_from_to2)
    def backward_D_to3(self):
        fake_from_to3 = self.fake_to3_pool.query(torch.cat((self.real_from, self.fake_to3), 1))
        real_from_to3 = torch.cat((self.real_from, self.real_to3), 1)
        self.loss_D_to3 = self.backward_D_basic(self.netD_to2, fake_from_to3, real_from_to3)
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_from_to1 = torch.cat((self.real_from, self.fake_to1), 1)
        fake_from_to2 = torch.cat((self.real_from, self.fake_to2), 1)
        fake_from_to3 = torch.cat((self.real_from, self.fake_to3), 1)
        fake_from_to1 = Variable(fake_from_to1.data, requires_grad=True)   #  - important
        fake_from_to2 = Variable(fake_from_to2.data, requires_grad=True)
        fake_from_to3 = Variable(fake_from_to3.data, requires_grad=True)
        pred_fake_to1 = self.netD_to1(fake_from_to1)   # 32 x 1 x 30 x 30
        pred_fake_to2 = self.netD_to2(fake_from_to2)
        pred_fake_to3 = self.netD_to3(fake_from_to3)
        self.loss_G_to1_GAN = self.criterionGAN(pred_fake_to1, True)
        self.loss_G_to2_GAN = self.criterionGAN(pred_fake_to2, True)
        self.loss_G_to3_GAN = self.criterionGAN(pred_fake_to3, True)

        # Second, G(A) = B
        fake_to1 = Variable(self.fake_to1.data, requires_grad=True)    #  - important
        fake_to2 = Variable(self.fake_to2.data, requires_grad=True)
        fake_to3 = Variable(self.fake_to3.data, requires_grad=True)
        self.loss_G_to1_L1 = self.criterionL1(fake_to1, self.real_to1) * self.opt.lambda_L1
        self.loss_G_to2_L1 = self.criterionL1(fake_to2, self.real_to2) * self.opt.lambda_L1
        self.loss_G_to3_L1 = self.criterionL1(fake_to3, self.real_to3) * self.opt.lambda_L1
        # Combined Loss
        self.loss_G_L1 = self.loss_G_to1_L1 + self.loss_G_to2_L1 + self.loss_G_to3_L1
        self.loss_G_GAN = self.loss_G_to1_GAN + self.loss_G_to2_GAN + self.loss_G_to3_GAN
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad([self.netD_to1, self.netD_to2, self.netD_to3], True)
        self.optimizer_D.zero_grad()
        self.backward_D_to1()
        self.backward_D_to2()
        self.backward_D_to3()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad([self.netD_to1, self.netD_to2, self.netD_to3], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

