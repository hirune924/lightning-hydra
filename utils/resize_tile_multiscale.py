import os

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
from tqdm import tqdm
import skimage.io
from skimage.transform import resize, rescale
from argparse import ArgumentParser
import shutil

import numpy as np

import openslide

from resize_intl_tile import load_img


def main(args):

    train_labels = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    for i in range(5,16):
        os.makedirs(os.path.join(args.save_dir, "train_images_"+str(i)), exist_ok=True)
    shutil.copyfile(
        os.path.join(args.data_dir, "sample_submission.csv"),
        os.path.join(args.save_dir, "sample_submission.csv"),
    )
    shutil.copyfile(
        os.path.join(args.data_dir, "train.csv"),
        os.path.join(args.save_dir, "train.csv"),
    )
    shutil.copyfile(
        os.path.join(args.data_dir, "test.csv"), os.path.join(args.save_dir, "test.csv")
    )

    for img_id in tqdm(train_labels.image_id):
        print(img_id)
        for i in range(5,16):
            load_path = os.path.join(args.data_dir, "train_images/" + img_id + ".tiff")
            save_path = os.path.join(args.save_dir, "train_images_" + str(i) + "/" + img_id + ".png")

            # biopsy = skimage.io.MultiImage(load_path)
            #biopsy_tile = load_img(load_path, K=16)
            biopsy_tile = load_img(load_path, K=16, scaling_factor=i*0.1, layer=0)
            img = cv2.resize(biopsy_tile, (args.size, args.size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img)

    """
    os.makedirs(os.path.join(args.save_dir, 'train_label_masks'), exist_ok=True)
    mask_files = os.listdir(os.path.join(args.data_dir, 'train_label_masks'))
    
    for mask_file in tqdm(mask_files):
        load_path = os.path.join(args.data_dir, 'train_label_masks/' + mask_file)
        save_path = os.path.join(args.save_dir, 'train_label_masks/' + mask_file.replace('.tiff', '.png'))

        #mask = skimage.io.MultiImage(load_path)[0]
        mask_tile = load_img(load_path, 0)
        img = cv2.resize(mask_tile, (args.size, args.size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)
    """


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--size", help="image size", type=int, required=False, default=256
    )
    parser.add_argument(
        "-sd", "--save_dir", help="path to log", type=str, required=True
    )
    parser.add_argument(
        "-dd", "--data_dir", help="path to data dir", type=str, required=True
    )

    # args = parser.parse_args(['-dd', '../input/prostate-cancer-grade-assessment/', '-sd','../working'])
    args = parser.parse_args()

    main(args)
