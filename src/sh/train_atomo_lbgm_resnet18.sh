cifar100(){
    python train_federated.py --device-id 2 0 --dataset cifar100 --clf resnet18 --paradigm atomo lbgm --error-tol 0.2 --atomo-r $k --optim sgd --num-workers 50 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-1 --momentum 0.9 --scheduler 0 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}

n=$2
k=$3

$1
