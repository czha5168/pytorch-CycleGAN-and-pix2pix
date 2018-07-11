# Created by Chaoyi.
# Aims for BraTS17 Dataset
# Learn the difference between B and B', with Ground Truth
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedBraTS17AllInOneDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print('opt.phase = ', opt.phase)
        #self.dir_ABCDE = os.path.join(opt.dataroot, opt.phase)
        self.dir_ABCDE = opt.dataroot
        self.ABCDE_paths = sorted(make_dataset(self.dir_ABCDE))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        ABCDE_path = self.ABCDE_paths[index]
        ABCDE = Image.open(ABCDE_path).convert('RGB')
        w, h = ABCDE.size
        w2 = int(w / 5)
        w3 = w2 * 2
        w4 = w2 * 3
        w5 = w2 * 4

        A = ABCDE.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = ABCDE.crop((w2, 0, w3, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        C = ABCDE.crop((w3, 0, w4, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        D = ABCDE.crop((w4, 0, w5, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        E = ABCDE.crop((w5, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        C = transforms.ToTensor()(C)
        D = transforms.ToTensor()(D)
        E = transforms.ToTensor()(E)

        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        C = C[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        D = D[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        E = E[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
        D = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(D)
        E = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(E)

        assert (self.opt.input_nc == self.opt.output_nc)
        '''
        # Both RGB so far
        input_nc = self.opt.output_nc
        output_nc = self.opt.input_nc
        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        '''


        #self.opt.which_direction == 'BtoA':

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            C = C.index_select(2, idx)
            D = D.index_select(2, idx)
            E = E.index_select(2, idx)

        return {'T1': A, 'T2': B, 'T1CE': C, 'FLAIR':D, 'SEG':E,
                'all_in_one_path': ABCDE_path}

    def __len__(self):
        return len(self.ABCDE_paths)

    def name(self):
        return 'AlignedBraTS17AllInOneDataset'
