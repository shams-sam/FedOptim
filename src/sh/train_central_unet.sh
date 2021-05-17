################################################################################
# coco - 0.5 subset: wip
################################################################################
coco(){
    python train_centralized.py --device-id 1 2 0 --dataset coco --clf unet --paradigm sgd --batch-size 512 --epochs 101 --lr 1e-4 --momentum 0.9 --repeat 0.5 --dry-run 0 --early-stopping 0
}


################################################################################
# voc
# config reference: https://github.com/yassouali/pytorch-segmentation
# unet architecture:
    # https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-\
    #for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862
################################################################################
voc(){
    python train_centralized.py --device-id 2 1 0 --dataset voc --clf unet --paradigm sgd --batch-size 32 --epochs 101 --lr 1e-4 --momentum 0.9 --repeat 0.5 --dry-run 0 --early-stopping 0
}



$1
