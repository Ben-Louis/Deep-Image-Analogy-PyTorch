import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import load_image
import argparse
from DeepAnalogy import analogy

def str2bool(v):
    return v.lower() in ('true')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--resize_ratio', type=float, default=0.5)
    parser.add_argument('--weight', type=int, default=2, choices=[2,3])
    parser.add_argument('--img_A_path', type=str, default='data/demo/ava.png')
    parser.add_argument('--img_BP_path', type=str, default='data/demo/mona.png')
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    args = parser.parse_args()


    # load images
    img_A = load_image(args.img_A_path, args.resize_ratio)
    img_BP = load_image(args.img_BP_path, args.resize_ratio)


    # setting parameters
    config = dict()

    params = {
        'layers': [29,20,11,6,1],
        'iter': 10,
    }
    config['params'] = params

    if args.weight == 2:
        config['weights'] = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
    elif args.weight == 3:
        config['weights'] = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
    config['sizes'] = [3,3,3,5,5,3]
    config['rangee'] = [32,6,6,4,4,2]

    config['use_cuda'] = args.use_cuda
    config['lr'] = [0.1, 0.005, 0.005, 0.00005]

    # Deep-Image-Analogy
    img_AP, img_B = analogy(img_A, img_BP, config)

    content = os.listdir('results')
    count = 1
    for c in content:
        if os.path.isdir('results/'+c):
            count += 1
    save_path = 'results/expr_{}'.format(count)
    os.mkdir(save_path)

    plt.imsave(save_path+'/img_AP.png', img_AP)
    plt.imsave(save_path+'/img_B.png', img_B)


    
    
    


