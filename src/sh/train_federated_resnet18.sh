################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 0 1 2 --dataset celeba --clf resnet18 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --repeat 0.4 --dry-run 0 &
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf resnet18 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid $n --repeat 1 --dry-run 0 --save-model 1 &
}

################################################################################
# fmnist: do for 3
################################################################################
fmnist(){
    python train_federated.py --device-id 0 2 --dataset fmnist --clf resnet18 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}

################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 0 2 1 --dataset mnist --clf resnet18 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --repeat 1 --dry-run 0 &
}


n=$2

$1