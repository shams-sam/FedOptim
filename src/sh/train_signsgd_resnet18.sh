################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 1 2 0 --dataset celeba --clf resnet18 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-4 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0 &
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 0 1 2 --dataset cifar --clf resnet18 --paradigm signsgd --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-3 --momentum 0.9 --scheduler 0 --non-iid $n --repeat 1 --dry-run 0 &
}


################################################################################
# cifar100
################################################################################
cifar100(){
    python train_federated.py --device-id 2 0 --dataset cifar100 --clf resnet18 --paradigm signsgd --optim sgd --num-workers 50 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-2 --momentum 0.9 --scheduler 0 --non-iid $n --repeat 1 --dry-run 0 &
}


################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf resnet18 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-4 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0
}


################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf resnet18 --paradigm signsgd --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-4 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0
}


n=$2

$1
