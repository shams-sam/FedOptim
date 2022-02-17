################################################################################
# celeba - 0.5 subset: done
################################################################################
celeba(){
    python train_centralized.py --device-id 2 0 --dataset celeba --clf resnet18 --paradigm sgd --batch-size 256 --epochs 101 --lr 1e-3 --momentum 0.9 --loss-type mse --repeat 0.5 --dry-run 0 --early-stopping 0
}


################################################################################
# cifar: done
################################################################################
cifar(){
    python train_centralized.py --device-id 1 2 0 --dataset cifar --clf resnet18 --paradigm sgd --batch-size 128 --epochs 201 --lr 1e-1 --momentum 0.9 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0 &
}


################################################################################
# cifar100: done
################################################################################
# 40: 0.1
# 10: 0.01 or 5: 0.02
cifar100(){
    python train_centralized.py --device-id 1 2 0 --dataset cifar100 --clf resnet18 --optim sgd --batch-size 128 --epochs 201 --lr 1e-1 --momentum 0.9 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0 --save-model 0 &
}


################################################################################
# fmnist: done
################################################################################
fmnist(){
    python train_centralized.py --device-id 2 1 0 --dataset fmnist --clf resnet18 --paradigm sgd --batch-size 256 --epochs 301 --lr 1e-2 --momentum 0.9 --repeat 1 --dry-run 0 --early-stopping 0 &
}

################################################################################
# miniimagenet
################################################################################
miniimagenet(){
    python train_centralized.py --device-id 1 2 0 --dataset miniimagenet --clf resnet18 --optim sgd --batch-size 128 --epochs 201 --lr 1e-3 --momentum 0.9 --scheduler 0 --repeat 1 --dry-run 0 --early-stopping 0 --save-model 0 &
}


################################################################################
# mnist: done
################################################################################
mnist(){
    python train_centralized.py --device-id 2 1 0 --dataset mnist --clf resnet18 --paradigm sgd --batch-size 256 --epochs 101 --lr 1e-2 --repeat 1 --dry-run 0 --early-stopping 0 &
}


$1
