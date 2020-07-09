# ILA-plus-plus
Code for our ECCV 2020 paper **Yet Another Intermediate-Level Attack**.

## Requirements
* Python 3.7
* Numpy 1.19.0
* Pillow 6.0.0
* PyTorch 1.3.0
* Torchvision 0.4.1

## Datasets
Select images from ImageNet validation set, and write ```.csv``` file as following:
```
class_index, class, image_name
0,n01440764,ILSVRC2012_val_00002138.JPEG
2,n01484850,ILSVRC2012_val_00004329.JPEG
...
```

## Usage
Full usage:
```
python attack.py -h
usage: attack.py [-h] [--batch_size BATCH_SIZE] [--epsilon EPSILON] [--lr LR]
                 [--baseline_niters BASELINE_NITERS] [--ila_niters ILA_NITERS]
                 [--mid_layer_index MID_LAYER_INDEX] [--std_ila] [--lam LAM]
                 [--lam_inf] [--normalize_H]
                 [--baseline_method BASELINE_METHOD] [--save_w]
                 [--skip_baseline_attack] [--w_dir W_DIR]
                 [--adv_save_dir ADV_SAVE_DIR] [--dataset_dir DATASET_DIR]
                 [--selected_images_csv SELECTED_IMAGES_CSV]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size
  --epsilon EPSILON     epsilon
  --lr LR               learning rate of both baseline attack and ila attack
  --baseline_niters BASELINE_NITERS
                        number of iterations of baseline attack
  --ila_niters ILA_NITERS
                        number of iterations of ila attack
  --mid_layer_index MID_LAYER_INDEX
                        intermediate layer index
  --std_ila             whether do standard ila attack, default is False
  --lam LAM             lambda
  --lam_inf             whether set lambda = infinity, default is True
  --normalize_H         whether normalize H, default is True
  --baseline_method BASELINE_METHOD
                        ifgsm/pgd/mifgsm_{momentum}(e.g. mifgsm_0.9)/tap
  --save_w              whether save w*, default is False
  --skip_baseline_attack
                        whether skip baseline attack, default is False
  --w_dir W_DIR         w* directory
  --adv_save_dir ADV_SAVE_DIR
                        adversarial examples directory
  --dataset_dir DATASET_DIR
                        ImageNet-val directory.
  --selected_images_csv SELECTED_IMAGES_CSV
                        path of the csv file of selected images.
```

## Acknowledgements
The following resources are very helpful for our work:

* [Pretrained models for ImageNet](https://github.com/Cadene/pretrained-models.pytorch)
* [Pretrained models for CIFAR-100](https://github.com/bearpaw/pytorch-classification)
* [GDAS](https://github.com/D-X-Y/GDAS)