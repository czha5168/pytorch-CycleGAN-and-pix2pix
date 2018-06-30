set -ex
python ./../test.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/t1-to-t2 --name t1-to-t2_pix2pix_maskGiven --model pix2pix_BraTS17 --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned_BraTS17 --norm batch
