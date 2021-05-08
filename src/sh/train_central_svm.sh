################################################################################
# cifar
################################################################################
cifar(){
    python train_centralized.py --device-id 1 2 0 --dataset cifar --clf svm --paradigm sgd --batch-size 60000 --epochs 201 --lr 1e-4 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0
    python train_centralized.py --device-id 1 2 0 --dataset cifar --clf svm --paradigm sgd --batch-size 128 --epochs 201 --lr 1e-4 --scheduler 1 --repeat 1 --dry-run 0 --early-stopping 0
}


$1
