import os
from utils import load_image
import argparse
from DeepAnalogy import analogy
import cv2

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resize_ratio', type=float, default=0.5)
    parser.add_argument('--weight', type=int, default=2, choices=[2,3])
    parser.add_argument('--img_A_path', type=str, default='data/demo/ava.png')
    parser.add_argument('--img_BP_path', type=str, default='data/demo/mona.png')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--use_cuda', action="store_true")

    config = parser.parse_args()
    if config.weight == 2:
        config.blend_weights = [0, 0.1, 0.6, 0.7, 0.8, 1.0]
    elif config.weight == 3:
        config.blend_weights = [0, 0.2, 0.7, 0.8, 0.9, 1.0]
    config.pm_iter = 10
    config.pm_sizes = [3, 5, 5, 3, 3, 3]
    config.pm_range = [2, 4, 4, 6, 6, 32]
    config.deconv_lr = [0, 0, 5e-5, 5e-3, 5e-3, 1e-1]

    return config


def main(config):
    # load images
    print('Loading images...', end='')
    img_A = load_image(config.img_A_path, config.resize_ratio)
    img_BP = load_image(config.img_BP_path, config.resize_ratio)
    print('\rImages loaded successfully!')

    # Deep-Image-Analogy
    print("##### \tDeep Image Analogy - start #####")
    img_AP, img_B, elapse = analogy(img_A, img_BP, config)
    print(f"##### \tDeep Image Analogy - end | Elapse: {elapse} #####")

    # save result
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    cv2.imwrite(f"{config.save_path}/img_AP.png", img_AP)
    cv2.imwrite(f"{config.save_path}/img_B.png", img_B)
    print('Image saved!')


if __name__=="__main__":
    main(get_parameters())