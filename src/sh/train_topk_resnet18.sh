################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 0 2 --dataset celeba --clf resnet18 --paradigm topk --topk $k --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 1 --repeat 0.4 --dry-run 0
}

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 1 0 2 --dataset cifar --clf resnet18 --paradigm topk --topk $k --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-2 --momentum 0.9 --scheduler 0 --non-iid $n --residual 1 --repeat 1 --dry-run 0
}


################################################################################
# cifar
################################################################################
cifar100(){
    python train_federated.py --device-id 1 0 2 --dataset cifar100 --clf resnet18 --paradigm topk --topk $k --optim sgd --num-workers 50 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-1 --momentum 0.9 --scheduler 0 --non-iid $n --residual 1 --repeat 1 --dry-run 0 &
}



################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf resnet18 --paradigm topk --topk $k --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 1 --repeat 1 --dry-run 0
}



################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf resnet18 --paradigm topk --topk $k --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 1 --repeat 1 --dry-run 0
}



n=$2
k=$3

$1
