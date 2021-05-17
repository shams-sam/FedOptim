################################################################################
# cifar: done
################################################################################
cifar(){
    python train_centralized.py --device-id 1 2 0 --dataset cifar --clf svm --paradigm sgd --batch-size 128 --epochs 201 --lr 1e-4 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0
}


################################################################################
# fmnist: done
################################################################################
fmnist(){
    python train_centralized.py --device-id 2 1 0 --dataset fmnist --clf svm --paradigm sgd --batch-size 128 --epochs 101 --lr 1e-4 --momentum 0.9 --repeat 1 --dry-run 0 --early-stopping 0
}


################################################################################
# mnist: done
################################################################################
mnist(){
    python train_centralized.py --device-id 2 1 0 --dataset mnist --clf svm --paradigm sgd --batch-size 128 --epochs 101 --lr 1e-4 --repeat 1 --dry-run 0 --early-stopping 0
}



$1
