################################################################################
# imagenet: done
################################################################################
imagenet(){
    python train_centralized.py --device-id 1 2 0 --dataset imagenet --clf resnet34 --paradigm sgd --batch-size 512 --epochs 101 --lr 1e-1 --momentum 0.9 --scheduler 1 --repeat 0.5 --dry-run 0 --early-stopping 0
}

################################################################################
# cifar100
################################################################################
cifar100(){
    python train_centralized.py --device-id 1 2 0 --dataset cifar100 --clf resnet101 --optim sgd --batch-size 128 --epochs 201 --lr 1e-1 --momentum 0.9 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0 --save-model 0 &
}

$1
