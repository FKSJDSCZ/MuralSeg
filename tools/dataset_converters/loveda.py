# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LoveDA dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='LoveDA folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'loveDA')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    for dataset in ['Train', 'Val', 'Test']:
        data_type = dataset.lower()
        for location in ['Rural', 'Urban']:
            for image_type in ['images_png', 'masks_png']:
                if image_type == 'images_png':
                    dst = osp.join(out_dir, 'img_dir', data_type)
                else:
                    dst = osp.join(out_dir, 'ann_dir', data_type)
                if dataset == 'Test' and image_type == 'masks_png':
                    continue
                else:
                    src_dir = osp.join(dataset_path, dataset, location,
                                       image_type)
                    src_lst = os.listdir(src_dir)
                    for file in src_lst:
                        shutil.copy2(osp.join(src_dir, file), dst)

    print('Done!')


if __name__ == '__main__':
    main()
