################################################################################
# celeba: running
################################################################################
celeba(){
    python train_federated.py --device-id 2 0 1 --dataset celeba --clf fcn --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset celeba --clf fcn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset celeba --clf fcn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset celeba --clf fcn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset celeba --clf fcn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 301 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0 &
}


################################################################################
# cifar: to do 10, 3
################################################################################
cifar(){
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf fcn --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset cifar --clf fcn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf fcn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf fcn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar --clf fcn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}

################################################################################
# cifar: to do 10, 3
################################################################################
cifar100(){
    python train_federated.py --device-id 2 1 0 --dataset cifar100 --clf fcn --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset cifar100 --clf fcn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar100 --clf fcn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar100 --clf fcn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 1 0 --dataset cifar100 --clf fcn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 501 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}

################################################################################
# fmnist: to do: 3
################################################################################
fmnist(){
    python train_federated.py --device-id 2 0 1 --dataset fmnist --clf fcn --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset fmnist --clf fcn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset fmnist --clf fcn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset fmnist --clf fcn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 2 0 1 --dataset fmnist --clf fcn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}

################################################################################
# mnist: to do: 10, 3
################################################################################
mnist(){
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf fcn --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf fcn --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf fcn --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf fcn --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
    python train_federated.py --device-id 0 1 2 --dataset mnist --clf fcn --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 100 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0 &
}

n=$2

$1
