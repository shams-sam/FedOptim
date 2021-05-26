################################################################################
# cifar
################################################################################
cifar(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf fcn --paradigm atomo --atomo-r 1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf fcn --paradigm atomo --atomo-r 2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 10 --repeat 1 --dry-run 0 &
}

cifar_residual(){
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf fcn --paradigm atomo --atomo-r 1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset cifar --clf fcn --paradigm atomo --atomo-r 2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 1e-2 --momentum 0.9 --scheduler 1 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
}



################################################################################
# fmnist:
################################################################################
fmnist(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --paradigm atomo --atomo-r 1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --paradigm atomo --atomo-r 2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
}

fmnist_residual(){
    python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --paradigm atomo lbgm --error-tol 0.8 --atomo-r 1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --residual 1 --repeat 1 --dry-run 1
    # python train_federated.py --device-id 1 2 0 --dataset fmnist --clf fcn --paradigm atomo lbgm --error-tol 0.8 --atomo-r 2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
}



################################################################################
# mnist:
################################################################################
mnist(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf fcn --paradigm atomo --atomo-r 1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf fcn --paradigm atomo --atomo-r 2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --repeat 1 --dry-run 0 &
}

mnist_residual(){
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf fcn --paradigm atomo --atomo-r 1 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 1 2 0 --dataset mnist --clf fcn --paradigm atomo --atomo-r 2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid 10 --residual 1 --repeat 1 --dry-run 0 &
}


$1
