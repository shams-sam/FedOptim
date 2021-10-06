celeba(){
    python train_federated.py --device-id 2 0 1 --dataset celeba --clf resnet18 --paradigm atomo lbgm --error-tol 0.4 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --test-type fl --loss-type mse --epochs 101 --lr 1e-5 --momentum 0.9 --scheduler 0 --non-iid $n --residual 0 --repeat 0.4 --dry-run 0 &
}

cifar(){
    python train_federated.py --device-id 2 0 1 --dataset cifar --clf resnet18 --paradigm atomo lbgm --error-tol 0.4 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-2 --momentum 0.9 --scheduler 0 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}


cifar100(){
    python train_federated.py --device-id 2 0 1 --dataset cifar100 --clf resnet18 --paradigm atomo lbgm --error-tol 0.6 --atomo-r $k --optim sgd --num-workers 50 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-1 --momentum 0.9 --scheduler 0 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}

n=$2
k=$3

$1
