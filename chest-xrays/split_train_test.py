import os
import shutil


train_fnames = [line.strip() for line in open('train_val_list.txt', 'r').readlines()]
test_fnames = [line.strip() for line in open('test_list.txt', 'r').readlines()]

all_folder = 'processed'

train_folder = 'train/subfolder'
test_folder = 'test/subfolder'

for fnames, folder in ([train_fnames, train_folder], [test_fnames, test_folder]):

    os.makedirs(folder)
    for fname in fnames:
        if os.path.exists(os.path.join(all_folder, fname)):  # TODO remove
            shutil.move(os.path.join(all_folder, fname), os.path.join(folder, fname))

assert len(os.listdir(all_folder)) == 0
os.rmdir(all_folder)
