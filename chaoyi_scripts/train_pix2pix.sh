set -ex
# Version1: Raw Code Running with BraTS2017 dataset

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/t1-to-t2 --name t1-to-t2_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 #[OriginalCommand]

##################################################################################################################################################

# Version2: Cross_Modalities
# Details: Two more visualization blocks including diff_B and Mask_C(new_pix2pix_model=pix2pix_BraTS17_CrossModalities and new_data_loader=aligned_BraTS17_dataset)

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/t1-to-t2 --display_ncols 5 --display_winsize 256 --name CrossModalityTransaction/t1-to-t2_pix2pix_maskGiven --model pix2pix_BraTS17_CrossModalities --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned_BraTS17 --niter 50 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[T1toT2]

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/t1-to-flair --display_ncols 5 --display_winsize 256 --name CrossModalityTransaction/t1-to-flair_pix2pix_maskGiven --model pix2pix_BraTS17_CrossModalities --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned_BraTS17 --niter 50 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[T1toFlair]

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/t2-to-t1 --display_ncols 5 --display_winsize 256 --name CrossModalityTransaction/t2-to-t1_pix2pix_maskGiven --model pix2pix_BraTS17_CrossModalities --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned_BraTS17 --niter 50 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[T2toT1]

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/t2-to-flair --display_ncols 5 --display_winsize 256 --name CrossModalityTransaction/t2-to-flair_pix2pix_maskGiven --model pix2pix_BraTS17_CrossModalities --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned_BraTS17 --niter 50 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[T2toFlair]


#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/flair-to-t1 --display_ncols 5 --display_winsize 256 --name CrossModalityTransaction/flair-to-t1_pix2pix_maskGiven --model pix2pix_BraTS17_CrossModalities --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned_BraTS17 --niter 50 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[FlairtoT1]

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/btw_modalities_with_segmask/flair-to-t2 --display_ncols 5 --display_winsize 256 --name CrossModalityTransaction/flair-to-t2_pix2pix_maskGiven --model pix2pix_BraTS17_CrossModalities --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned_BraTS17 --niter 50 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[FlairtoT2]
##################################################################################################################################################

# Version3: SingleModality_to_SegMask
# Details: Use pix2pix to achieve segmentation from single modality to the segMask

#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/reorder/train/shuffleSplit_imgs_for_pix2pix/map_to_segmask/flair-to-seg --display_ncols 5 --display_winsize 256 --name flair_to_segMask_pix2pix --model pix2pix_BraTS17_SingleModalitySegMask --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 # learn to segment from Flair

##################################################################################################################################################


# Version4: OneModalToThreeModal
# implment: unet_256_multiple_outputs 
#		- Reimplement it as Encoder-Decoder format
# 		- Improve it to Multiple Outputs Version
#	    aligned_BraTS17_all_in_one
#		- Dataloader to load T1/T2/T1ce/flair/SegMask at once
#	    pix2pix_BraTS17_MultipleOutputs_model
#		- Employ unet_256_multiple_outputs as its generator
#		- Create three seperate discriminator to deal with each branch output seperately
#		- Loss = (G_L1 + G_GAN + D_real + D_fake) * (from_to1, from_to2, from_to3)
#	    new dataset format = train_all_in_one_flatten_all_cases
#python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/info_kept/train_all_in_one_flatten_50_cases --display_ncols 2 --display_winsize 256 --name OneEncoderThreeDecoder/T1 --model pix2pix_BraTS17_MultipleOutputs --which_model_netG unet_256_multiple_outputs --which_direction FROM_T1 --lambda_L1 100 --dataset_mode aligned_BraTS17_all_in_one --batchSize 16 --niter 500 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[FlairtoT1]

python ./../train.py --dataroot /media/machine/Storage/Dataset/BraTS17_MICCAI/sliceExtracted_2D/info_kept/train_all_in_one_flatten_all_cases --display_ncols 2 --display_winsize 256 --name OneEncoderThreeDecoder/T1 --model pix2pix_BraTS17_MultipleOutputs --which_model_netG unet_256_multiple_outputs --which_direction FROM_T1 --lambda_L1 100 --dataset_mode aligned_BraTS17_all_in_one --batchSize 16 --niter 500 --no_lsgan --norm batch --pool_size 0 #[UpdatedCommand-MaskReferenceGiven&Diff_B_computed] #[FlairtoT1]
