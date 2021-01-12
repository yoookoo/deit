import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imnet-root', type=str)
    parser.add_argument('--imfolder-root', type=str, default='./imagenet_folder')
    args = parser.parse_args()

    if not os.path.isdir(args.imfolder_root):
        os.makedirs(args.imfolder_root)

    # create imangenet class folder
    for i in range(1000):
        train_cls_folder = os.path.join(args.imfolder_root, 'train', str(i))
        val_cls_folder = os.path.join(args.imfolder_root, 'val', str(i))
        if not os.path.isdir(train_cls_folder):
            os.makedirs(train_cls_folder)
        if not os.path.isdir(val_cls_folder):
            os.makedirs(val_cls_folder)

    # fetch image meta files
    train_meta = open(os.path.join(args.imnet_root, 'meta', 'train.txt'), 'r')
    val_meta = open(os.path.join(args.imnet_root, 'meta', 'val.txt'), 'r')

    total_train = 0
    for l in train_meta.readlines():
        fname, label = l.strip().split(' ')
        file_path = os.path.join(args.imnet_root, 'train', *fname.split('/'))
        symlink_path = os.path.join(args.imfolder_root, 'train', label, fname.split('/')[1])
        if not os.path.exists(symlink_path):
            os.symlink(file_path, symlink_path)
        total_train += 1

    total_val = 0
    for l in val_meta.readlines():
        fname, label = l.strip().split(' ')
        file_path = os.path.join(args.imnet_root, 'val', fname)
        symlink_path = os.path.join(args.imfolder_root, 'val', label, fname)
        if not os.path.exists(symlink_path):
            os.symlink(file_path, symlink_path)
        total_val += 1

    print('done! total train data: {}, total val data: {}'.format(total_train, total_val))
