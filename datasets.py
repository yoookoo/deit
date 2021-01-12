# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import mc
import io
import logging
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image

logger = logging.getLogger('global')


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class ImageNetDataSet(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(ImageNetDataSet, self).__init__()

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.initialized = False

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __len__(self):
        return self.num

    def read_file(self, meta_dict):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(meta_dict['filename'], value)
        value_str = mc.ConvertBuffer(value)
        return np.frombuffer(value_str.tobytes(), dtype=np.uint8)

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = os.path.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        img_bytes = self.read_file(curr_meta)
        img = pil_loader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if is_train:
            root_dir = os.path.join(args.data_path, 'train')
            meta_file = os.path.join(args.data_path, 'meta', 'train.txt')
        else:
            root_dir = os.path.join(args.data_path, 'val')
            meta_file = os.path.join(args.data_path, 'meta', 'val.txt')
        dataset = ImageNetDataSet(root_dir, meta_file, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
