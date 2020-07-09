import os
import torch
import time
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import random
import argparse
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.backends import cudnn
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=200, help='batch size ')
parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon')
parser.add_argument('--lr', type=float, default=1./255, help='learning rate of both baseline attack and ila attack')
parser.add_argument('--baseline_niters', type=int, default=10, help='number of iterations of baseline attack')
parser.add_argument('--ila_niters', type=int, default=50, help='number of iterations of ila attack')
parser.add_argument('--mid_layer_index', type=str, default='3_1', help='intermediate layer index')
parser.add_argument('--std_ila', default=False, action='store_true', help='whether do standard ila attack, default is False')
parser.add_argument('--lam', type=float, default=1.0, help='lambda')
parser.add_argument('--lam_inf', default=True, action='store_false', help='whether set lambda = infinity, default is True')
parser.add_argument('--normalize_H', default=True, action='store_false', help='whether normalize H, default is True')
parser.add_argument('--baseline_method', type=str, default='ifgsm', help='ifgsm/pgd/mifgsm_{momentum}(e.g. mifgsm_0.9)/tap')
parser.add_argument('--save_w', default=False, action='store_true', help='whether save w*, default is False')
parser.add_argument('--skip_baseline_attack', default=False, action='store_true', help='whether skip baseline attack, default is False')
parser.add_argument('--w_dir', type=str, default='./results/w/imagenet', help='w* directory')
parser.add_argument('--adv_save_dir', type=str, default='./results/adversarial/imagenet', help='adversarial examples directory')
parser.add_argument('--dataset_dir', type=str, default='./dataset/ILSVRC2012_img_val', help='ImageNet-val directory.')
parser.add_argument('--selected_images_csv', type=str, default='./dataset/selected_imagenet.csv', help='path of the csv file of selected images.')
args = parser.parse_args()





if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    epsilon = args.epsilon
    baseline_niters = args.baseline_niters
    ila_niters = args.ila_niters
    batch_size = args.batch_size
    std_ila = args.std_ila
    lam = args.lam
    if args.lam_inf:
        lam = 'inf'
    normalize_H = args.normalize_H
    mid_layer_index = args.mid_layer_index
    bi, ui = mid_layer_index.split('_')
    mid_layer_index = '{}_{}'.format(bi, int(ui)-1)
    baseline_method = args.baseline_method
    lr = args.lr
    w_dir = args.w_dir
    skip_baseline_attack = args.skip_baseline_attack
    save_w = args.save_w
    if skip_baseline_attack:
        save_w = False
    adv_save_dir = args.adv_save_dir
    dataset_dir = args.dataset_dir
    selected_images_csv = args.selected_images_csv
    if save_w:
        os.makedirs(w_dir, exist_ok=True)
    os.makedirs(adv_save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # set dataset & dataloader
    trans = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor()
    ])
    dataset = SelectedImagenet(imagenet_val_dir=dataset_dir,
                               selected_images_csv=selected_images_csv,
                               transform=trans
                               )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)


    # set source model
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    model = nn.Sequential(
            Normalize('imagenet'),
            model
        )
    model = model.to(device)

    # attack
    for ind, (ori_img, label)in enumerate(dataloader):
        if skip_baseline_attack:
            w = np.load(w_dir + '/w_{}.npy'.format(ind))
        else:
            H, r = attack(False, None, ori_img, label, device, baseline_niters, baseline_method, epsilon, model, mid_layer_index, batch_size, lr)
            if std_ila:
                w = H[:,-1,:]
            else:
                w = calculate_w(H=H, r=r, lam=lam, normalize_H = normalize_H)
            if save_w:
                np.save(w_dir + '/w_{}.npy'.format(ind), w)
        attacked_imgs = attack(True, torch.from_numpy(w), ori_img, label, device, ila_niters, baseline_method, epsilon, model, mid_layer_index, batch_size, lr)
        np.save(adv_save_dir + '/{}.npy'.format(ind), attacked_imgs.cpu().numpy())
        print('{} adversarial images have been saved.'.format(ind*batch_size+attacked_imgs.size(0)))