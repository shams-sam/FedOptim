################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 1 2 0 --dataset celeba --clf vgg19 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 0 1 2 --dataset cifar --clf vgg19 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-4 --momentum 0.9 --scheduler 1 --non-iid $n --repeat 1 --dry-run 0
}


################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf vgg19 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-4 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0
}


################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf vgg19 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-4 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0
}


n=$2

$1
