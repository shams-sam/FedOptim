################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 0 --dataset cifar --clf vgg19 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
}


################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 0 --dataset fmnist --clf vgg19 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
}

################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 0 --dataset mnist --clf vgg19 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0
}



$1
