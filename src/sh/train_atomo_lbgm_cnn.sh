################################################################################
# celeba
################################################################################
celeba(){
    python train_federated.py --device-id 1 2 0 --dataset celeba --clf cnn --paradigm atomo lbgm  --error-tol 0.4 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 1e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0 &
    }

################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf cnn --paradigm atomo lbgm --error-tol 0.2 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --scheduler 0 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}


################################################################################
# cifar100
################################################################################
cifar100(){
    python train_federated.py --device-id 1 2 0 --dataset cifar100 --clf cnn --paradigm atomo lbgm --error-tol 0.2 --atomo-r $k --optim sgd --num-workers 50 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 501 --lr 1e-1 --momentum 0.9 --scheduler 0 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}


################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --paradigm atomo lbgm --error-tol 0.2 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf cnn --paradigm atomo lbgm --error-tol 0.4 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}


################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 2 0 --dataset mnist --clf cnn --paradigm atomo lbgm --error-tol 0.4 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 --dataset mnist --clf cnn --paradigm atomo lbgm --error-tol 0.2 --atomo-r $k --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}



n=$2
k=$3

$1
