################################################################################
# celeba:
################################################################################
celeba(){
    # python train_federated.py --device-id 1 0 --dataset celeba --clf vgg19 --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0
    python train_federated.py --device-id 1 0 --dataset celeba --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 101 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0
    # python train_federated.py --device-id 1 0 --dataset celeba --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0
    # python train_federated.py --device-id 1 0 --dataset celeba --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0
    # python train_federated.py --device-id 1 0 --dataset celeba --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --epochs 201 --lr 2e-5 --momentum 0.9 --loss-type mse --non-iid 0 --residual 0 --repeat 0.4 --dry-run 0
}


################################################################################
# cifar: 3 do all; 10 do 0.2
################################################################################
cifar(){
    # python train_federated.py --device-id 2 0 1 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    python train_federated.py --device-id 2 0 1 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 2 0 1 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 2 0 1 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 2 0 1 --dataset cifar --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 201 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
}

################################################################################
# fmnist: running 0.9 for fmnist 10
################################################################################
fmnist(){
    # python train_federated.py --device-id 0 1  --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    python train_federated.py --device-id 0 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 0 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 0 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 0 1 --dataset fmnist --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
}

################################################################################
# mnist: 10 do 0.9 ; 3 do all
################################################################################
mnist(){
    # python train_federated.py --device-id 2 1 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.9 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    python train_federated.py --device-id 2 1 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.8 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 2 1 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.6 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 2 1 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.4 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
    # python train_federated.py --device-id 2 1 0 --dataset mnist --clf vgg19 --paradigm lbgm --error-tol 0.2 --optim sgd --num-workers 10 --batch-size 0 --test-batch-size 128 --test-type fl --epochs 101 --lr 1e-2 --momentum 0.9 --non-iid $n --residual 0 --repeat 1 --dry-run 0
}

n=$2

$1
