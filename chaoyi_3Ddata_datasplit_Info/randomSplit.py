import random
from collections import defaultdict
from glob import glob
# Official Training Dataset
train_glob = '/media/machine/Storage/Dataset/BraTS17_MICCAI/raw_3D/MICCAI_BraTS17_Data_Training/*/*/*.nii.gz'
train_files = glob(train_glob)
random.shuffle(train_files)
temp_map = defaultdict(list)
for path in train_files:
    tokens = path.split('/')
    tumor_grade = tokens[-3]
    case_idx = tokens[-2]
    file_idx = tokens[-1]
    print(tumor_grade, case_idx, file_idx)

# Official Validation Dataset
valid_glob = '/media/machine/Storage/Dataset/BraTS17_MICCAI/raw_3D/Brats17ValidationData/*/*/*.nii.gz'
valid_files = glob(valid_glob)
print(train_files)
